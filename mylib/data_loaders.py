import torch
import torch.utils.data as data


def get_data_loader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    """Vanilla data loader from PyTorch
    """
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)
