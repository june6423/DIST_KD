import torch.nn as nn
import torch

def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)
    

def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


def inter_class_relation_spearman(y_s, y_t):
    argmax_y_s = torch.argmax(y_s, dim=1)
    argmax_y_t = torch.argmax(y_t, dim=1)
    
    y_s_rank = torch.argsort(y_s, dim=1)
    y_t_rank = torch.argsort(y_t, dim=1)
   
    y_s_rank = torch.clamp(y_s_rank, max=10)
    y_t_rank = torch.clamp(y_t_rank, max=10)
   
    y_s_filtered = torch.cat([
        y_s_rank[i, :j].unsqueeze(0) if j == y_s_rank.shape[1] - 1 else torch.cat([y_s_rank[i, :j], y_s_rank[i, j+1:]], dim=0).unsqueeze(0)
        for i, j in enumerate(argmax_y_s)
    ], dim=0).float()
    y_t_filtered = torch.cat([
        y_t_rank[i, :j].unsqueeze(0) if j == y_t_rank.shape[1] - 1 else torch.cat([y_t_rank[i, :j], y_t_rank[i, j+1:]], dim=0).unsqueeze(0)
        for i, j in enumerate(argmax_y_t)
    ], dim=0).float()
    
    spearman_rank_correlation = pearson_correlation(y_s_filtered, y_t_filtered).mean()
    
    equal = (argmax_y_s == argmax_y_t).float().mean()

    return 1 - (equal + spearman_rank_correlation)

class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss

class DIST_plus(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0):
        super(DIST_plus, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation_spearman(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss