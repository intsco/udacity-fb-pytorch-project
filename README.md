# Udacity PyTorch Scholarship Challenge Project

## Introduction
Deep learning for multilabel image classification.

The details about the challenge can be found on its
[website](https://sites.google.com/udacity.com/pytorch-scholarship-facebook/phase-1-archived/phase-1-home)

As the final project for the Challenge, classification of 102 flower species from
[Visual Geometry Group University of Oxford dataset]((http://www.robots.ox.ac.uk/~vgg/data/flowers/102)
was chosen.

The dataset consists of 102 flower categories. The flowers chosen to be flower commonly occuring in the United Kingdom. Each class consists of between 40 and 258 images of approximate size 500x600 pixels.

The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories.

Unofficial project leader board on [Kaggle](https://www.kaggle.com/c/oxford-102-flower-pytorch/leaderboard).

## Methods
I used ResNet architectures of various sizes.
Different learning rate schedulers, like
[OneCycleLR](https://arxiv.org/abs/1803.09820),
[CyclicLR](https://arxiv.org/pdf/1506.01186.pdf), and
[WarmRestartsLR](https://arxiv.org/abs/1608.03983) were explored.

A modified version of Adam optimizer
([AdamW](https://www.fast.ai/2018/07/02/adam-weight-decay/))
was used to train the networks.

The standard ResNet models from PyTorch were modifed using a set of "tricks" inspired by
[fast.ai library](https://github.com/fastai/fastai)

Additionally, the use of TTA (test time augmentation) and combination of
multiple model trained using cross validation, made it possible
to achieve top performance even with a relatively small ResNet34 model.

On top of that, training of such a model took less than 20 minutes on Tesla K80 GPU.

## Results
To reach **99.5%** accuracy on the test dataset, I used 5 Resnet101 models trained using cross validation.
But it was possible to achieve **97%** accuracy even with smaller networks like ResNet34.

