import torch

def get_generator(seed, frame_index):
    return torch.Generator("cpu").manual_seed(seed + frame_index)
