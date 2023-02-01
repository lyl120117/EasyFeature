import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


class CosineAnnealing(object):

    def __init__(self, T_max, eta_min=1e-6):
        self.T_max = T_max
        self.eta_min = eta_min

    def __call__(self, optimizer):
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=self.T_max,
                                      eta_min=self.eta_min)
        return scheduler


class MultiStep(object):

    def __init__(self, milestones, gamma=0.1):
        self.milestones = milestones
        self.gamma = gamma

    def __call__(self, optimizer):
        scheduler = MultiStepLR(optimizer,
                                milestones=self.milestones,
                                gamma=self.gamma)
        return scheduler


if __name__ == '__main__':
    step = 100
    # optimizer = torch.optim.Adam([torch.randn(1, 1)], lr=0.1)
    # scheduler = CosineAnnealing(step)(optimizer)
    optimizer = torch.optim.SGD([torch.randn(1, 1)], lr=0.1)
    scheduler = MultiStep([50, 65])(optimizer)

    for i in range(step):
        optimizer.step()
        scheduler.step()

        lr1 = scheduler.get_last_lr()[0]
        lr2 = optimizer.param_groups[0]['lr']
        print(i, lr1, lr2)