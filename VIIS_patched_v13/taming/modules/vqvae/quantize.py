
import torch, torch.nn as nn
from einops import rearrange
class VectorQuantizer2(nn.Module):
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        assert remap is None, "remap not supported in this stub"
        self.n_e, self.e_dim, self.beta, self.legacy = n_e, e_dim, beta, legacy
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)
        self.remap = None
        self.re_embed = n_e
        self.sane_index_shape = sane_index_shape
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        zf = z.view(-1, self.e_dim)
        d = (zf ** 2).sum(1, keepdim=True) + (self.embedding.weight ** 2).sum(1) \
            - 2 * zf @ self.embedding.weight.t()
        idx = torch.argmin(d, dim=1)
        zq = self.embedding(idx).view(z.shape)
        if not self.legacy:
            loss = self.beta * ((zq.detach() - z) ** 2).mean() + ((zq - z.detach()) ** 2).mean()
        else:
            loss = ((zq.detach() - z) ** 2).mean() + self.beta * ((zq - z.detach()) ** 2).mean()
        zq = z + (zq - z).detach()
        zq = rearrange(zq, "b h w c -> b c h w").contiguous()
        if self.sane_index_shape:
            idx = idx.reshape(zq.shape[0], zq.shape[2], zq.shape[3])
        return zq, loss, (None, None, idx)
    def get_codebook_entry(self, indices, shape):
        zq = self.embedding(indices)
        if shape is not None:
            zq = zq.view(shape).permute(0, 3, 1, 2).contiguous()
        return zq
