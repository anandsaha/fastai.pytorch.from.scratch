import torch
from tqdm import *
import torch.autograd as autograd


def show(opt):
    print('Here')
    for group in opt.param_groups:
        for p in group['params']:
            print('Param', type(p))
            if p.grad is not None:
                print('Grad', type(p.grad.data))


# With this function, we can experiment with various parameter combinations
def train1(model,
           criteria,
           optimizer,
           exp_lr_scheduler,
           train_loader,
           valid_loader,
           num_epochs=5,
           debug=False
           ):
    if debug:
        print('Total training instances:', len(train_loader.dataset))
        print('Total validation instances:', len(valid_loader.dataset))
        print('Classes:', train_loader.dataset.classes)

    # We will track the best validation precision available, and the checkpoint file it is contained in
    best_accuracy = 0.0
    best_loss = 1.0
    best_epoch = 0
    best_model_weights = None

    train_loss_trend = list()
    val_loss_trend = list()

    for e in range(num_epochs):

        print('\nEpoch {}/{}'.format(e + 1, num_epochs))

        #################################################
        # Train
        #################################################

        model.train()
        exp_lr_scheduler.step()

        train_loss = 0.0
        train_accuracy = 0.0

        for i, (input, target) in enumerate(train_loader):
            input_var = autograd.Variable(input.cuda())
            target_var = autograd.Variable(target.cuda())

            optimizer.zero_grad()
            output = model(input_var).cuda()
            _, preds = torch.max(output.data, 1)

            loss = criteria(output, target_var)

            loss.backward()
            if i == 0:
                show(optimizer)
            optimizer.step()

            train_loss += loss.data[0]
            train_accuracy += torch.sum(preds == target_var.data)
            train_loss_trend.append(loss.cpu().data.numpy()[0])

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_accuracy / len(train_loader.dataset)

        #################################################
        # Validate
        #################################################
        # Every epoch we calculate the validation accuracy

        model.eval()

        val_loss = 0.0
        val_accuracy = 0.0
        count = 0

        for i, (input, target) in tqdm(enumerate(valid_loader), leave=True):
            input_var = autograd.Variable(input.cuda())
            target_var = autograd.Variable(target.cuda())
            # optimizer.zero_grad()
            output = model(input_var).cuda()
            _, preds = torch.max(output.data, 1)
            loss = criteria(output, target_var)

            val_loss += loss.data[0]
            val_accuracy += torch.sum(preds == target_var.data)
            val_loss_trend.append(loss.cpu().data.numpy()[0])

        epoch_val_loss = val_loss / len(valid_loader.dataset)
        epoch_val_acc = val_accuracy / len(valid_loader.dataset)

        print('Training Loss  :', epoch_train_loss, ', Acc:', epoch_train_acc)
        print('Validation Loss:', epoch_val_loss, ', Acc:', epoch_val_acc)

        #################################################
        # Checkpoint
        #################################################
        # Save the model if best so far
        if epoch_val_acc > best_accuracy:
            print('############################### Better model found')
            best_model_weights = model.state_dict()
            best_accuracy = epoch_val_acc
            best_loss = epoch_val_loss
            best_epoch = e

    model.load_state_dict(best_model_weights)

    return model, train_loss_trend, val_loss_trend, best_epoch, best_accuracy, best_loss


# With this function, we can experiment with various parameter combinations
def train(model,
          criteria,
          optimizer,
          exp_lr_scheduler,
          train_loader,
          valid_loader,
          num_epochs=5,
          debug=False
          ):
    if debug:
        print('Total training instances:', len(train_loader.dataset))
        print('Total validation instances:', len(valid_loader.dataset))
        print('Classes:', train_loader.dataset.classes)

    # We will track the best validation precision available, and the checkpoint file it is contained in
    best_accuracy = 0.0
    best_loss = 1.0
    best_epoch = 0
    best_model_weights = None

    train_loss_trend = list()
    val_loss_trend = list()

    for e in tnrange(num_epochs, desc='Epoch'):

        print('\nEpoch {}/{}'.format(e + 1, num_epochs))

        #################################################
        # Train
        #################################################

        model.train()
        exp_lr_scheduler.step()

        train_loss = 0.0
        train_accuracy = 0.0

        for i, (input, target) in tqdm(enumerate(train_loader), leave=True):
            input_var = autograd.Variable(input.cuda())
            target_var = autograd.Variable(target.cuda())

            optimizer.zero_grad()
            output = model(input_var).cuda()
            _, preds = torch.max(output.data, 1)

            loss = criteria(output, target_var)

            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            train_accuracy += torch.sum(preds == target_var.data)
            train_loss_trend.append(loss.cpu().data.numpy()[0])

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_accuracy / len(train_loader.dataset)

        #################################################
        # Validate
        #################################################
        # Every epoch we calculate the validation accuracy

        model.eval()

        val_loss = 0.0
        val_accuracy = 0.0
        count = 0

        for i, (input, target) in tqdm(enumerate(valid_loader), leave=True):
            input_var = autograd.Variable(input.cuda())
            target_var = autograd.Variable(target.cuda())
            # optimizer.zero_grad()
            output = model(input_var).cuda()
            _, preds = torch.max(output.data, 1)
            loss = criteria(output, target_var)

            val_loss += loss.data[0]
            val_accuracy += torch.sum(preds == target_var.data)
            val_loss_trend.append(loss.cpu().data.numpy()[0])

        epoch_val_loss = val_loss / len(valid_loader.dataset)
        epoch_val_acc = val_accuracy / len(valid_loader.dataset)

        print('Training Loss  :', epoch_train_loss, ', Acc:', epoch_train_acc)
        print('Validation Loss:', epoch_val_loss, ', Acc:', epoch_val_acc)

        #################################################
        # Checkpoint
        #################################################
        # Save the model if best so far
        if epoch_val_acc > best_accuracy:
            print('############################### Better model found')
            best_model_weights = model.state_dict()
            best_accuracy = epoch_val_acc
            best_loss = epoch_val_loss
            best_epoch = e

    model.load_state_dict(best_model_weights)

    return model, train_loss_trend, val_loss_trend, best_epoch, best_accuracy, best_loss
