import torchvision.models as models
import torch.nn as nn
import torch
import torch.autograd


family_alexnet = 'alexnet'
family_densenet = 'densenet'
family_resnet = 'resnet'
family_inception = 'inception'
family_squeezenet = 'squeezenet'
family_vgg = 'vgg'

input_sizes = {
    'alexnet' : (224,224),
    'densenet': (224,224),
    'resnet' : (224,224),
    'inception' : (299,299),
    'squeezenet' : (224,224),
    'vgg' : (224,224)
}


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def cond_init(m, init_fn):
    if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if hasattr(m, 'weight'): init_fn(m.weight)
        if hasattr(m, 'bias'): m.bias.data.fill_(0.)


def apply_init(m, init_fn):
    m.apply(lambda x: cond_init(x, init_fn))


def model_resnet_vanilla(num_classes):
    model = models.resnet34(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # Reconfigure the last layer _only_ to adjust to the classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.cuda()
    return model


def resnet34_extra_layers(num_classes,
                          top_layers_to_freeze=8,
                          fc_layers_dim=[512, 512],
                          dropout_p=[0.25, 0.25],
                          final_dropout_p=0.5,
                          debug=False):
    model = models.resnet34(pretrained=True)
    resnet34_layers_to_extract = 8

    layers_list = list(model.children())
    feature_extracting_layers = layers_list[:resnet34_layers_to_extract]

    feature_extracting_layers += [AdaptiveConcatPool2d(), Flatten()]
    # top_model = nn.Sequential(*layers)

    fc_layers = []

    in_dim = 1024  # TODO - Find this dynamically

    # Add the requested FC layers
    for idx, out_dim in enumerate(fc_layers_dim):
        fc = [nn.BatchNorm1d(in_dim),
              nn.Linear(in_features=in_dim, out_features=out_dim),
              nn.Dropout(dropout_p[idx]),
              nn.ReLU()]
        fc_layers += fc
        in_dim = out_dim

    # Add the output layers
    fc = [nn.BatchNorm1d(in_dim),
          nn.Dropout(final_dropout_p),
          nn.Linear(in_features=in_dim, out_features=num_classes),
          nn.LogSoftmax()]
    fc_layers += fc

    # Compose the conv layers and fc layers
    model = nn.Sequential(*(feature_extracting_layers + fc_layers))
    model = model.cuda()

    for param in model.parameters():
        param.requires_grad = False

    layer_num = 0
    for c in model.children():
        if layer_num >= top_layers_to_freeze:
            for param in c.parameters():
                param.requires_grad = True
        layer_num += 1

    if debug:
        for c in model.children():
            param_count = 0
            param_trainable = 0
            for p in c.parameters():
                param_count += 1
                if p.requires_grad:
                    param_trainable += 1
            print(type(c), param_count, param_trainable)

    layer_num = 0
    params_to_optimize = list()
    for c in model.children():
        if layer_num >= top_layers_to_freeze:
            params_to_optimize.append({'params': c.parameters()})
        layer_num += 1

    # Time to freeze stuff

    return model, params_to_optimize


def inception3_extra_layers(num_classes,
                            top_layers_to_freeze=15,
                            fc_layers_dim=[1024],
                            dropout_p=[0.5],
                            final_dropout_p=0.5,
                            debug=False):
    model = models.inception_v3(pretrained=True)
    inception3_layers_to_extract = 17

    layers_list = list(model.children())
    feature_extracting_layers = layers_list[:inception3_layers_to_extract]

    feature_extracting_layers += [AdaptiveConcatPool2d(), Flatten()]

    fc_layers = []

    in_dim = 1024  # TODO - Find this dynamically

    # Add the requested FC layers
    for idx, out_dim in enumerate(fc_layers_dim):
        fc = [nn.BatchNorm1d(in_dim),
              nn.Linear(in_features=in_dim, out_features=out_dim),
              nn.Dropout(dropout_p[idx]),
              nn.ReLU()]
        fc_layers += fc
        in_dim = out_dim

    # Add the output layers
    fc = [nn.BatchNorm1d(in_dim),
          nn.Dropout(final_dropout_p),
          nn.Linear(in_features=in_dim, out_features=num_classes),
          nn.LogSoftmax()]
    fc_layers += fc

    # Compose the conv layers and fc layers
    model = nn.Sequential(*(feature_extracting_layers + fc_layers))
    model = model.cuda()

    for param in model.parameters():
        param.requires_grad = False

    layer_num = 0
    for c in model.children():
        if layer_num >= top_layers_to_freeze:
            for param in c.parameters():
                param.requires_grad = True
        layer_num += 1

    if debug:
        for c in model.children():
            param_count = 0
            param_trainable = 0
            for p in c.parameters():
                param_count += 1
                if p.requires_grad:
                    param_trainable += 1
            print(type(c), param_count, param_trainable)

    layer_num = 0
    params_to_optimize = list()
    for c in model.children():
        if layer_num >= top_layers_to_freeze:
            params_to_optimize.append({'params': c.parameters()})
        layer_num += 1

    # Time to freeze stuff

    return model, params_to_optimize


def inception3_extra_layers_test(num_classes,
                                 top_layers_to_freeze=15,
                                 fc_layers_dim=[1024],
                                 dropout_p=[0.5],
                                 final_dropout_p=0.5,
                                 debug=False):
    model = models.inception_v3(pretrained=True, transform_input=True)
    model.fc = nn.Linear(2048, num_classes)
    #layers_list = list(model.children())
    #feature_extracting_layers = layers_list[:17]
    #feature_extracting_layers += [nn.Linear(2048, num_classes)]
    #model = nn.Sequential(*(feature_extracting_layers))

    model = model.cuda()

    for param in model.parameters():
        param.requires_grad = False

    layer_num = 0
    for c in model.children():
        if layer_num >= top_layers_to_freeze:
            for param in c.parameters():
                param.requires_grad = True
        layer_num += 1

    if debug:
        for c in model.children():
            param_count = 0
            param_trainable = 0
            for p in c.parameters():
                param_count += 1
                if p.requires_grad:
                    param_trainable += 1
            print(type(c), param_count, param_trainable)

    layer_num = 0
    params_to_optimize = list()
    for c in model.children():
        if layer_num >= top_layers_to_freeze:
            params_to_optimize.append({'params': c.parameters()})
        layer_num += 1

    # Time to freeze stuff

    return model, params_to_optimize
