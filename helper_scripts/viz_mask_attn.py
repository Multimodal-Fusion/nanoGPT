import numpy as np
import matplotlib.pyplot as plt
import torch


# mask function
def mask_mod(q_idx, kv_idx, t, r):
    # identify real/register rows/cols
    is_real_q  = q_idx < t
    is_real_k  = kv_idx < t
    k_in_regs  = (kv_idx >= t) & (kv_idx < t + r * t)
    # parent token index of q (works for both real and register q)
    parent_i = torch.where(is_real_q, q_idx, (q_idx - t) % t)
    # kv is a register of parent_i  <=>  kv in regs AND (kv - t - parent_i) % t == 0
    same_token_reg = k_in_regs & (((kv_idx - t - parent_i) % t) == 0)
    # permissions
    allow_real_q_real_k = is_real_q & is_real_k & (kv_idx <= q_idx)   # causal among reals
    allow_real_q_regs   = is_real_q & same_token_reg                  # real -> its registers
    allow_reg_q_parent  = (~is_real_q) & (kv_idx == parent_i)         # register -> parent real
    allow_reg_q_regs    = (~is_real_q) & same_token_reg               # register -> its registers
    return allow_real_q_real_k | allow_real_q_regs | allow_reg_q_parent | allow_reg_q_regs


t = 3
r=2

# create a grid of q_idx and kv_idx
# we loop q_idx and kv_idx from 0 to r*t-1
mask = torch.zeros((r*t, r*t))
# create a list of q_idx and kv_idx as torch tensors
q_idx_list = list(range(r*t))
kv_idx_list = list(range(r*t))
# create a tensor of q_idx and kv_idx
q_idx_tensor = torch.tensor(q_idx_list)
kv_idx_tensor = torch.tensor(kv_idx_list)
# create a tensor of mask
mask_tensor = mask_mod(q_idx_tensor, kv_idx_tensor, t, r)
print(mask_tensor)

# convert to numpy
mask = mask.numpy()

# save the mask as a png image
plt.imshow(mask)
plt.savefig('mask.png')