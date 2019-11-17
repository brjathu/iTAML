''' Incremental-Classifier Learning
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from resnet import *

class BasicNet1(nn.Module):

    def __init__(
        self, args, use_bias=False, init="kaiming", use_multi_fc=False, device=None
    ):
        super(BasicNet1, self).__init__()

        self.use_bias = use_bias
        self.init = init
        self.use_multi_fc = use_multi_fc
        self.args = args

        if(self.args.dataset=="mnist"):
            self.convnet = RPS_net_mlp()
        elif(self.args.dataset=="svhn"):
            self.convnet = RPS_net(self.args.num_class)
        elif(self.args.dataset=="cifar100"):
            self.convnet = RPS_net(self.args.num_class)
        elif(self.args.dataset=="omniglot"):
            self.convnet = RPS_net(self.args.num_class)
        elif(self.args.dataset=="celeb"): 
            self.convnet = resnet18()
    
        
        self.classifier = None

        self.n_classes = 0
        self.device = device
        self.cuda()
        
    def forward(self, x):
        x1, x2 = self.convnet(x)
        return x1, x2

    @property
    def features_dim(self):
        return self.convnet.out_dim

    def extract(self, x):
        return self.convnet(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def copy(self):
        return copy.deepcopy(self)
    
    def add_classes(self, n_classes):
        if self.use_multi_fc:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes
        
    def _add_classes_multi_fc(self, n_classes):
        if self.classifier is None:
            self.classifier = []

        new_classifier = self._gen_classifier(n_classes)
        name = "_clf_{}".format(len(self.classifier))
        self.__setattr__(name, new_classifier)
        self.classifier.append(name)

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.n_classes + n_classes)

        if self.classifier is not None:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, n_classes):
#         torch.manual_seed(self.seed)
        classifier = nn.Linear(self.convnet.out_dim, n_classes, bias=self.use_bias).cuda()
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        if self.use_bias:
            nn.init.constant_(classifier.bias, 0.)

        return classifier

    
    
    