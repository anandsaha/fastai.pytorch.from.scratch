# Big Picture



# Data

### Image Data

**Miscellaneous**

* If using pre-trained weights, images need to be normalized in the _exact_ same way as was done to the original images used during pre-training. (For PyTorch pre-trained models, details are [here](http://pytorch.org/docs/master/torchvision/models.html). Fastai takes care of that in [transforms.py](https://github.com/fastai/fastai/blob/master/fastai/transforms.py))
* Image sizes used are typically 224 or 299. In fastai, size is to be passed to the transformer like so: `tfms_from_model(resnet34, sz=224)`
* _Sometimes_ progressively increasing the image size as you train may give better accuracy (e.g. start with 224 and switch to 299 along the way)
* For classification task, images might be arranged in directories of classes, or the class mapping might be in a csv file or mappings might be loaded in a numpy array. In fastai, you can use `ImageClassifierData.from_paths()` or `ImageClassifierData.from_csv()` or `ImageClassifierData.from_arrays()` to read in these images.

**Data Augmentation**

* We achieve better generalization by applying transformations to our existing data to bring variety in the way images are seen by our model.
* Note that data augmentation do not _add_ to your dataset. If you have 10k images, you still train on 10k images. It's only that when these images are passed to the model, either the original is passed or a randomly transformed one passed. Over multiple epochs, for any given image, the model gets to see it's original form as well as it's transformed forms (statistically speaking).
* Example of transformations are Random Cropping, Random Scalling, Horizontal flipping etc.
* Most models assume that the input image be square in shape. If not, some libraries might squeeze them to make them square. It's better to have a transformation which crops a square instead of squeezing them, since that distorts the image. This is automatically done in fastai (CropType.RANDOM). 
* In fastai, use `tfms_from_model()` to create a `Transforms` object, like so:
```
tfms = tfms_from_model(resnet34, sz) # No transformation (except mandatory ones like normalization and cropping)
tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO) # Do not crop
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
tfms = tfms_from_model(resnet34, sz, aug_tfms=[RandomFlipXY(), RandomLightingXY(0.05, 0.05)])
```
* Be mindful of the transformations you use. For e.g., applying random vertical flip to MNIST might backfire since `6` and `9` become indistinguishable.
 
_(Also See Test Time Augmentation in Inferencing section)_

### Text Data

_TODO_






# Model Design

### Image Classification Models

* Model selection: Start with a simple model like resnet34. Make sure you are getting encouraging results. Check [this advice](http://forums.fast.ai/t/how-to-pick-the-right-pretrained-model/8481/2).
* ResNet family:
* Inception family:
* 

### Non Linearities

### Optimizers

### Loss Functions








# Model Training

### Learning Rate

* A challenge to decide what learning rate to use
* fastai has a handy feature to find a good learning rate to start with, to be used like so:

```
learn = ConvLearner.pretrained(arch, data, precompute=True)
lrf=learn.lr_find()
learn.sched.plot()
```

From the above plot, pick the largest learning rate that is still getting better. Refer these blog posts for more on the topic:[by @bushaev](https://techburst.io/improving-the-way-we-work-with-learning-rate-5e99554f163b).


### Epochs, Cycles, Cycle Len, Cycle Mult

* Start with a few epochs and see the trend

### Batch Size

* Batch size is limited by the amount of GPU memory you have.
* _Typically_ if you increase the batch size for some reason, the learning rate also needs an increase. Do an `lr_find()` to check.
* If your error plot is fluctuating a lot, it might be due to a smaller batch size.
* Batch size is set as a parameter called `bs` and passed to the `ImageClassifierData.from_xxxx()` family of functions.

### Weights Initialization

### Pre-Computing activations

### Transfer Learning considerations

**Tuning Learning Rate**

**Using Weight Decay**

### Hyper Parameters to tune





# Inferencing

* In fastai, `learner.predict()` will return predictions on the validation set that was passed during training. This will return the log of the predictions. Apply `exp()` to get the probabilities.

# Hardware and Setup

# Concepts

* Visualizing images and image kernels: http://setosa.io/ev/image-kernels/
* Universal Approximation Theorem: http://neuralnetworksanddeeplearning.com/chap4.html



# Papers

* Visualizing and Understanding Convolutional Networks: https://arxiv.org/abs/1311.2901
* Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186 (the `lr_find()` functionality is inspired by this work)


