# iTAML : An Incremental Task-Agnostic Meta-learning Approach
Official implementation of "iTAML : An Incremental Task-Agnostic Meta-learning Approach". (CVPR 2020) [(paper link)](http://papers.nips.cc/paper/9429-random-path-selection-for-continual-learning). 


This code provides an implementation for iTAML : An Incremental Task-Agnostic Meta-learning Approach (accepted at Conference on Computer Vision and Pattern Recognition, Seattle, Washington, 2020). This repository is implemented using pytorch and it includes code for running the incremental learning domain experiments on MNIST, SVHN, CIFAR100, ImageNet and MS-Celeb-10K.


### Dependencies
This code requires the following:
*matplotlib==3.2.1
*numpy==1.18.2
*pandas==1.0.3
*Pillow==7.0.0
*scipy==1.4.1
*torch==1.4.0
*torchvision==0.5.0

run `pip3 install -r requirements.txt` to install all the dependencies. 

### Data
For the Omniglot and MiniImagenet data, see the usage instructions in `data/omniglot_resized/resize_images.py` and `data/miniImagenet/proc_images.py` respectively.

### Usage
To run the code, see the usage instructions at the top of `main.py`.

### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/brjathu/iTAML/issues).

