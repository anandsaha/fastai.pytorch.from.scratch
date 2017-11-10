# fastai.pytorch.from.scratch
I am implementing from scratch the tools, techniques and best practices I learnt from fast.ai's 2017 offering.

The idea is to _not_ use the [fastai](https://github.com/fastai/fastai/tree/master/fastai) library and build everything **minimally** on PyTorch, to see things happening first hand.

I have referred code from the following sources (the exact source is also mentioned in each file):

* [The fastai library](https://github.com/fastai/fastai/tree/master/fastai)
* [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar/)
* [PyTorch examples](https://github.com/pytorch/examples)
* [PyTorch tutorials](http://pytorch.org/tutorials/)

Code tested on PyTorch v0.2.0, Python 3.6

### Data Loading

- [ ] Load from folders segregated into classes
- [ ] Load from csv files assigning classes to images


### Data Augmentation

- [ ] Horizontal flip
- [ ] Cropping center
- [ ] Cropping random
- [ ] Cropping custom
- [ ] Scalling

### Learning Rate

- [ ] Optimum learning rate finder
- [ ] LR Annealing

### Model Training

- [ ] Using pretrained weights
- [ ] Freezing layers
- [ ] Precompute activations
- [ ] Delete/Add Layers


### Test time techniques

- [ ] Test Time Augmentation (TTA)


### Papers

These papers or reading material was suggested either by Jeremy or other participants in various discussions.

* [GDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
* [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
* [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
* [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489)
* [Using convolutional networks and satellite imagery to identify patterns in urban environments at a large scale](https://arxiv.org/abs/1704.02965)
