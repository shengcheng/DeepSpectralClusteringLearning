import torch

class SpectralCLusterLayer(object):
    def inv_H(self, H_prev):
        H_prev_inv = H_prev.t()
        d = H_prev_inv.sum(1).unsqueeze(1).expand_as(H_prev_inv)
        d[d == 0] = 1
        return H_prev_inv / d


    def pseudo_inverse(self, X):
        u, s, v = torch.svd(X)
        h = torch.max(s) * float(max(X.size(0), X.size(1))) * 1e-15
        indices = torch.ge(s, h)
        indices2 = indices.eq(0)
        s[indices] = 1.0 / s[indices]
        s[indices2] = 0
        return torch.mm(torch.mm(v, torch.diag(s)), u.t())


    def grad_F(self, F, H):
        inv_F = self.pseudo_inverse(F)
        return torch.mm(torch.mm(F, torch.mm(inv_F, H)) - H, torch.mm(self.inv_H(H), inv_F.t()))