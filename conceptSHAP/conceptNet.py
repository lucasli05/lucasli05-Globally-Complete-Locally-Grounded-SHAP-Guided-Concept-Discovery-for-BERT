import torch
import torch.nn as nn
from itertools import chain, combinations
import numpy as np
import math

class ConceptNet(nn.Module):

    def __init__(self, n_concepts, train_embeddings):
        super(ConceptNet, self).__init__()
        embedding_dim = train_embeddings.shape[1]
        # random init using uniform dist
        self.concept = nn.Parameter(self.init_concept(embedding_dim, n_concepts), requires_grad=True)
        self.n_concepts = n_concepts
        self.train_embeddings = train_embeddings.transpose(0, 1) # (dim, all_data_size)

    def init_concept(self, embedding_dim, n_concepts):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concepts) + r_1
        return concept

    def forward(self, train_embedding, h_x, topk):
        """
        train_embedding: shape (bs, embedding_dim)
        """
        # calculating projection of train_embedding onto the concept vector space
        proj_matrix = (self.concept @ torch.inverse((self.concept.T @ self.concept))) \
                      @ self.concept.T # (embedding_dim x embedding_dim)
        proj = proj_matrix @ train_embedding.T  # (embedding_dim x batch_size)

        # passing projected activations through rest of model
        y_pred = h_x(proj.T)
        orig_pred = h_x(train_embedding)

        # Calculate the regularization terms as in new version of paper
        k = topk # this is a tunable parameter

        ### calculate first regularization term, to be maximized
        # 1. find the top k nearest neighbour
        all_concept_knns = []
        for concept_idx in range(self.n_concepts):
            c = self.concept[:, concept_idx].unsqueeze(dim=1) # (activation_dim, 1)

            # euc dist
            distance = torch.norm(self.train_embeddings - c, dim=0) # (num_total_activations)
            knn = distance.topk(k, largest=False)
            indices = knn.indices # (k)
            knn_activations = self.train_embeddings[:, indices] # (activation_dim, k)
            all_concept_knns.append(knn_activations)

        # 2. calculate the avg dot product for each concept with each of its knn
        L_sparse_1_new = 0.0
        for concept_idx in range(self.n_concepts):
            c = self.concept[:, concept_idx].unsqueeze(dim=1) # (activation_dim, 1)
            c_knn = all_concept_knns[concept_idx] # knn for c
            dot_prod = torch.sum(c * c_knn) / k # avg dot product on knn
            L_sparse_1_new += dot_prod
        L_sparse_1_new = L_sparse_1_new / self.n_concepts

        ### calculate Second regularization term, to be minimized
        all_concept_dot = self.concept.T @ self.concept
        mask = torch.eye(self.n_concepts).cuda() * -1 + 1 # mask the i==j positions
        L_sparse_2_new = torch.mean(all_concept_dot * mask)

        norm_metrics = torch.mean(all_concept_dot * torch.eye(self.n_concepts).cuda())

        return orig_pred, y_pred, L_sparse_1_new, L_sparse_2_new, [norm_metrics]

    ### MODIFICATION START: add shap_cluster_loss
    def shap_cluster_loss(self, cluster_centers):
        """
        cluster_centers: shape (n_clusters, embedding_dim) in np or torch tensor
        计算 sum_{cluster} min_{concept_i} Dist(concept_i, cluster).
        """
        if isinstance(cluster_centers, np.ndarray):
            cluster_centers = torch.from_numpy(cluster_centers).float().to(self.concept.device)
        
        # self.concept shape: (embedding_dim, n_concepts)
        # transpose => (n_concepts, embedding_dim)
        concept_mat = self.concept.T  # shape=(n_concepts, embedding_dim)
        n_clusters = cluster_centers.shape[0]

        # pairwise distance => shape=(n_clusters, n_concepts)
        diff = cluster_centers.unsqueeze(1) - concept_mat.unsqueeze(0)
        dist_sq = torch.sum(diff*diff, dim=2)
        dist = torch.sqrt(dist_sq + 1e-8)

        # for each cluster, min_{concept_i} dist
        min_dist, _ = torch.min(dist, dim=1)  # shape=(n_clusters,)
        shap_loss = torch.mean(min_dist)
        return shap_loss
    ### MODIFICATION END

    ### MODIFICATION START: add optional cluster_centers & lambda_shap
    def loss(self, train_embedding, train_y_true, h_x, regularize, doConceptSHAP, l_1, l_2, topk,
             cluster_centers=None, lambda_shap=0.03):
        """
        计算最终loss(预测+正则), completeness等
        如果 cluster_centers 和 lambda_shap>0, 加上 shap_loss
        """
        # Note: it is important to MAKE SURE L2 GOES DOWN! that will let concepts separate from each other

        orig_pred, y_pred, L_sparse_1_new, L_sparse_2_new, metrics = self.forward(train_embedding, h_x, topk)

        ce_loss = nn.CrossEntropyLoss()
        loss_new = ce_loss(y_pred, train_y_true)
        pred_loss = torch.mean(loss_new)

        # completeness score
        def n(y_pred):
            orig_correct = torch.sum(train_y_true == torch.argmax(orig_pred, axis=1))
            new_correct = torch.sum(train_y_true == torch.argmax(y_pred, axis=1))
            return torch.div(new_correct - (1/self.n_concepts), orig_correct - (1/self.n_concepts))

        completeness = n(y_pred)

        conceptSHAP = []
        if doConceptSHAP:
            def proj(concept):
                proj_matrix = (concept @ torch.inverse((concept.T @ concept))) \
                              @ concept.T  # (embedding_dim x embedding_dim)
                p_out = proj_matrix @ train_embedding.T  # (embedding_dim x batch_size)
                return h_x(p_out.T)

            c_id = np.arange(self.concept.shape[1])
            for idx in c_id:
                exclude = np.delete(c_id, idx)
                subsets = np.asarray(list(self.powerset(list(exclude))))
                sum_val = 0
                for subset in subsets:
                    c1 = list(subset) + [idx]
                    concept = np.take(self.concept.T.detach().cpu().numpy(), np.asarray(c1), axis=0)
                    concept = torch.from_numpy(concept).T.to(self.concept.device)
                    score1 = n(proj(concept))

                    if len(subset) > 0:
                        concept2 = np.take(self.concept.T.detach().cpu().numpy(), np.asarray(subset), axis=0)
                        concept2 = torch.from_numpy(concept2).T.to(self.concept.device)
                        score2 = n(proj(concept2))
                    else:
                        score2 = torch.tensor(0.)

                    norm_ = (math.factorial(len(c_id)-len(subset)-1)*math.factorial(len(subset))) / math.factorial(len(c_id))
                    sum_val += norm_ * (score1.item() - score2.item())
                conceptSHAP.append(sum_val)

        if regularize:
            final_loss = pred_loss + (l_1 * L_sparse_1_new * -1) + (l_2 * L_sparse_2_new)
        else:
            final_loss = pred_loss

        # Add shap-cluster if requested
        shap_loss_val = torch.tensor(0., device=train_embedding.device)
        if (cluster_centers is not None) and (lambda_shap > 0):
            shap_loss_val = self.shap_cluster_loss(cluster_centers)
            final_loss = final_loss + lambda_shap * shap_loss_val

        return completeness, conceptSHAP, final_loss, pred_loss, L_sparse_1_new, L_sparse_2_new, (metrics, shap_loss_val)
    ### MODIFICATION END

    def powerset(self, iterable):
        "powerset([1,2,3]) --> [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]"
        s = list(iterable)
        pset = chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))
        return [list(i) for i in list(pset)]
