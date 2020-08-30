import torch

class GradReverse(torch.autograd.Function):
    '''GRL layer from https://arxiv.org/abs/1409.7495'''

    @staticmethod
    def forward(ctx, x, lambd=0.5):
        # ctx is a context object that can be used to stash information
        # for backward computation,like intermediate parameters
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # We return the negated input multiplied by lambd
        # None is the backward for the lambd argument
        return ctx.lambd * grad_output.neg(), None

def grad_reverse(x, lambd=0.5):
    return GradReverse.apply(x, lambd)