import torch
import time


def save_checkpoint(epoch,
                    arch,
                    model,
                    val_accuracy,
                    val_loss,
                    optimizer,
                    filename_prefix):
    state = {
        'epoch': epoch + 1,
        'arch': arch,
        'model_state': model.state_dict(),
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'optimizer_state': optimizer.state_dict(),
    }

    filename = filename_prefix + "_" + str(time.time()) + ".pth.tar"
    torch.save(state, filename)
    return filename


def load_checkpoint(filename):

    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    arch = checkpoint['arch']
    model_state = checkpoint['model_state']
    val_accuracy = checkpoint['val_accuracy']
    val_loss = checkpoint['val_loss']
    optimizer_state = checkpoint['optimizer_state']
    return epoch, arch, model_state, val_accuracy, val_loss, optimizer_state
