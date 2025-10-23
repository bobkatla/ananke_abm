
import torch
def time_total_variation(U):
    diff = U[:,1:,:] - U[:,:-1,:]
    return torch.mean(torch.abs(diff))
