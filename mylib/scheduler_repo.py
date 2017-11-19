from torch.optim import lr_scheduler


def step_lr(optimizer,
            step_size=5,
            gamma=0.1):
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return exp_lr_scheduler
