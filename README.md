# Harmonic Network

My implementation of model introduced in [this paper](https://arxiv.org/abs/1812.03205v1).

I used PyTorch to implement pretty much all of the work.

The dataset I was using is the [skin cancer MNIST](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) from Kaggle.

I really enjoyed working on this project, I will try to fix some errors it has.

# Running

No models in here are pre-trained. Also, I have not (yet) been able to properly test the scripts for training and testing the model, so this is also on the to-do list.

## To do:

* Test on HAM10000, CIFAR10(?)
* Argument Parsing
* Add alpha-rooting enhancement
* Add some selection of a single color channel out of 3 (based on quality)
