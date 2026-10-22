
import torch, torch.nn.functional as F
def ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H*W for H, W in shapes], dim=1)
    grids = 2 * sampling_locations - 1
    out_l = []
    for lid, (H, W) in enumerate(shapes):
        v = value_list[lid].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H, W)
        g = grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        out_l.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    aw = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    out = (torch.stack(out_l, dim=-2).flatten(-2) * aw).sum(-1).view(N_, M_*D_, Lq_)
    return out.transpose(1, 2).contiguous()
class MSDeformAttnFunction:
    @staticmethod
    def apply(value, shapes, level_start, locs, w, im2col_step):
        return ms_deform_attn_core_pytorch(value, shapes.tolist(), locs, w)
