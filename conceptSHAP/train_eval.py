#!/usr/bin/env python
# coding: utf-8

"""
Full pipeline code for:
1) Loading IMDB embeddings + data
2) Possibly do SHAP analysis (sentence-level) + cluster => cluster_centers
3) Train ConceptNet with shap_cluster_loss if cluster_centers is present
4) Evaluate/interpret
"""

import torch
import numpy as np
import pandas as pd
import argparse
import math
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from tensorboardX import SummaryWriter

# ===== import the updated ConceptNet from conceptNet.py (unchanged structure) =====
from conceptNet import ConceptNet

# ===== interpretConcepts as your original code: plot_embeddings, save_concepts, concept_analysis
from interpretConcepts import plot_embeddings, save_concepts, concept_analysis

# ===== for shap cluster =====
import shap
from sklearn.cluster import KMeans
from transformers import BertForSequenceClassification, BertTokenizer


def extract_shap_topk_embeddings_and_cluster(
    model_bert,
    tokenizer,
    train_texts,
    train_embeddings,
    device,
    sample_size=800,
    n_clusters=5
):
    """
    使用 SHAP 分析 'sample_size' 条训练文本 -> sentence-level shap
    将其embedding再做 KMeans => cluster_centers
    (由于是“句子级”，我们实际上并没有 top_k token 这一说，但下面做个简单处理。)
    """
    def predict_proba(texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = ["" if t is None else str(t) for t in texts]

        enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            outputs = model_bert(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    print(f"[SHAP] Building Explainer, sample_size={sample_size}, n_clusters={n_clusters}")
    masker = shap.maskers.Text(tokenizer=tokenizer, mask_token=tokenizer.mask_token)
    explainer = shap.Explainer(predict_proba, masker=masker)

    used_size = min(sample_size, len(train_texts))
    texts_sub = train_texts[:used_size]
    print(f"[SHAP] Running shap on {used_size} texts ...")
    shap_values = explainer(texts_sub)  # might be slow

    # 这里 for sentence-level, shap_values[i].values => shape=(seq_len,2) typically
    # 但我们只要把“这个句子embedding”放进 cluster. 
    # 于是 top_token_embeddings ~> top_sentence_embeddings
    # top_sentence_embeddings = []
    # for i in range(used_size):
    #     # naive approach: we sum the absolute shap for label=1
    #     arr = shap_values[i].values
    #     if arr.ndim == 2:
    #         # (seq_len, 2)
    #         sum_val = np.sum(np.abs(arr[:,1]))
    #     else:
    #         # (seq_len,) maybe
    #         sum_val = np.sum(np.abs(arr))

    #     # sum_val 只是表示句子的 shap 大小, 
    #     # 但这里我们还是把embedding收集, 以备聚类
    #     # user might prefer picking top <some> shap samples, but let's do naive => all
    #     top_sentence_embeddings.append(train_embeddings[i])
        
    top_sentence_importance = []

    # 先计算每个句子的SHAP重要性
    for i in range(used_size):
        arr = shap_values[i].values
        if arr.ndim == 2:
            sum_val = np.sum(np.abs(arr[:, 1]))
        else:
            sum_val = np.sum(np.abs(arr))
    
        top_sentence_importance.append((sum_val, train_embeddings[i], texts_sub[i]))
    
    # 按照SHAP值降序排序
    top_sentence_importance.sort(reverse=True, key=lambda x: x[0])
    
    # 根据排序后SHAP重要性选择top-k句子embedding
    top_k = 80  # 假设你要前50个
    top_sentence_embeddings = [emb for _, emb, _ in top_sentence_importance[:top_k]]


    top_sentence_embeddings = np.array(top_sentence_embeddings)  # shape=(used_size, embed_dim)
    if len(top_sentence_embeddings) < n_clusters:
        print("[SHAP] Not enough embeddings to do KMeans, skip shap cluster!")
        return None

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(top_sentence_embeddings)
    cluster_centers = kmeans.cluster_centers_
    print("[SHAP] cluster_centers shape:", cluster_centers.shape)
    return cluster_centers


def train(args, train_embeddings, train_y_true, h_x, n_concepts, writer, device,
          cluster_centers=None, lambda_shap=0.003):
    '''
    The original training function with minimal modifications to pass cluster_centers & lambda_shap
    '''
    l_1 = args.l1
    l_2 = args.l2
    topk = args.topk
    batch_size = args.batch_size
    epochs = args.num_epochs
    cal_interval = args.shapley_interval

    # convert to torch
    train_embeddings = torch.from_numpy(train_embeddings).float().to(device)
    train_y_true = torch.from_numpy(train_y_true.astype('int64')).to(device)

    # freeze final layer
    for p in list(h_x.parameters()):
        p.requires_grad = False

    model = ConceptNet(n_concepts, train_embeddings).to(device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_size = train_embeddings.shape[0]
    loss_reg_epoch = args.loss_reg_epoch

    n_iter = 0
    for i in tqdm(range(epochs), desc="ConceptNet training"):
        if i < loss_reg_epoch:
            regularize = False
        else:
            regularize = True

        batch_start = 0
        batch_end = batch_size

        # shuffle
        train_y_true_float = train_y_true.float().unsqueeze(dim=1)
        data_pair = torch.cat([train_embeddings, train_y_true_float], dim=1)
        new_permute = torch.randperm(data_pair.shape[0])
        data_pair = data_pair[new_permute]
        permuted_train_embeddings = data_pair[:, :-1]
        permuted_train_y_true = data_pair[:, -1].long()

        while batch_start < train_size:
            end_idx = min(batch_end, train_size)
            batch_emb = permuted_train_embeddings[batch_start:end_idx]
            batch_lbl = permuted_train_y_true[batch_start:end_idx]

            do_shap = False
            completeness, conceptSHAP, final_loss, pred_loss, l1_val, l2_val, metrics = \
                model.loss(batch_emb, batch_lbl, h_x,
                           regularize=regularize,
                           doConceptSHAP=do_shap,
                           l_1=l_1,
                           l_2=l_2,
                           topk=topk,
                           cluster_centers=cluster_centers,   # pass if available
                           lambda_shap=lambda_shap
                           )

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            writer.add_scalar('sum_loss', final_loss.item(), n_iter)
            writer.add_scalar('pred_loss', pred_loss.item(), n_iter)
            writer.add_scalar('L1', l1_val.item(), n_iter)
            writer.add_scalar('L2', l2_val.item(), n_iter)
            # writer.add_scalar('norm_metrics', metrics[0].item(), n_iter)
            # if len(metrics) > 1:
            #     writer.add_scalar('shap_loss', metrics[1].item(), n_iter)

            writer.add_scalar('concept completeness', completeness.item(), n_iter)
            if conceptSHAP != []:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                plt.bar(list(range(len(conceptSHAP))), conceptSHAP)
                writer.add_figure('conceptSHAP', fig, n_iter)

            batch_start += batch_size
            batch_end += batch_size
            n_iter += 1

    return model, None  # not returning aggregated losses for now

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--activation_dir", type=str, required=True,
                        help="path to .npy file containing dataset embeddings")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="path to .pkl file containing train dataset")
    parser.add_argument("--bert_weights", type=str, required=True,
                        help="path to BERT config & weights directory")
    parser.add_argument("--n_concepts", type=int, default=5,
                        help="number of concepts to generate")

    parser.add_argument('--save_dir', default='./experiments',
                        help='directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='directory to save the log')
    parser.add_argument('--l1', type=float, default=.001)
    parser.add_argument('--l2', type=float, default=.002)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss_reg_epoch', type=int, default=2,
                        help="num of epochs to run without loss regularization")
    parser.add_argument('--num_epochs', type=int, default=3,
                        help="num of training epochs")
    parser.add_argument('--shapley_interval', type=int, default=5)

    # extra shap cluster
    parser.add_argument('--use_shap_cluster', action='store_true',
                        help="whether to do shap+cluster in the code")
    parser.add_argument('--shap_sample_size', type=int, default=2000,
                        help="samples used in shap")
    parser.add_argument('--shap_n_clusters', type=int, default=10,
                        help="clusters for shap-based cluster")
    parser.add_argument('--lambda_shap', type=float, default=0.03,
                        help="coefficient for shap cluster reg, 0=off")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    print("Loading dataset embeddings from", args.activation_dir)
    small_activations = np.load(args.activation_dir)
    print("Shape:", small_activations.shape)

    print("Loading dataset from", args.train_dir)
    data_frame = pd.read_pickle(args.train_dir)
    senti_list = np.array(data_frame['polarity'])

    print("Loading model weights from", args.bert_weights)
    bert_model = BertForSequenceClassification.from_pretrained(args.bert_weights)
    bert_model.to(device)

    # we also need a tokenizer for SHAP
    tokenizer = BertTokenizer.from_pretrained(args.bert_weights, do_lower_case=True)
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


    # h_x => final classification layer
    # or use: h_x = bert_model.classifier
    h_x = list(bert_model.modules())[-1]

    # we have train_embeddings + train_labels
    train_embeddings = small_activations
    train_labels = senti_list

    # step: optionally do SHAP + cluster
    # cluster_centers = None
    # if args.use_shap_cluster:
        # need textual data
    train_texts = []
    for i in range(len(data_frame)):
        s = data_frame['sentence'].iloc[i]
        if isinstance(s,list):
            s = " ".join(s)
        train_texts.append(str(s))

    cluster_centers = extract_shap_topk_embeddings_and_cluster(
        model_bert=bert_model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_embeddings=train_embeddings,  # shape=(n_samples, embed_dim)
        device=device,
        sample_size=args.shap_sample_size,
        n_clusters=args.shap_n_clusters
    )
    if cluster_centers is not None:
        print("Obtained cluster_centers from shap:", cluster_centers.shape)
    else:
        print("Shap cluster returned None => skip cluster regularization")
    
    # now train concept net
    from conceptNet import ConceptNet
    concept_model, _ = train(
        args,
        train_embeddings,
        train_labels,
        h_x,
        args.n_concepts,
        writer,
        device,
        cluster_centers=cluster_centers,
        lambda_shap=args.lambda_shap
    )

    # saving concepts
    save_concepts(concept_model)

    # do interpret
    plot_embeddings(small_activations, data_frame, senti_list, writer)
    concept_analysis(small_activations, data_frame)

    writer.close()
    print("All done.")
