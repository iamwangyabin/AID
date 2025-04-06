import torch


def inversion_first(x, seq, model):
    with torch.no_grad():
        n = x.size(0)
        t = (torch.ones(n) * seq[0]).to(x.device)
        et = model(x, t)

    return et

def norm(x):
    return (x - x.min()) / (x.max() - x.min())