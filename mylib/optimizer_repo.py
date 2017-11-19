import torch.nn as nn
import torch


def sgd(model,
        init_lr,
        params_to_optimize,
        momentum=0.9,
        weight_decay=1e-4):

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(params_to_optimize, lr=init_lr, momentum=momentum,
                                weight_decay=weight_decay)
    return criterion, optimizer


def rmsprop(model,
            init_lr,
            params_to_optimize):

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.RMSprop(params_to_optimize, lr=init_lr)
    return criterion, optimizer


def adam(model,
         init_lr,
         params_to_optimize):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(params_to_optimize, lr=init_lr)
    return criterion, optimizer
