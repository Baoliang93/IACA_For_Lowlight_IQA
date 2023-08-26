import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.autograd import Variable

def SPSP(x,xmax, net,P=1, method='avg'):
    batch_size = x.size(0)
    chan_size =  x.size(1)
    map_size = x.size()[-2:]
    pool_features = []
    x_max = F.adaptive_max_pool2d(xmax, map_size)
    x_max = net(x_max).repeat(1,chan_size,1,1)
    x = x_max*x
   
    for p in range(1, P+1):
        pool_size = [np.int(d / p) for d in map_size]
        if method == 'maxmin':
            M = F.max_pool2d(x, pool_size)
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(torch.cat((M, m), 1).view(batch_size, -1))  # max & min pooling
        elif method == 'max':
            M = F.max_pool2d(x, pool_size)
            pool_features.append(M.view(batch_size, -1))  # max pooling
        elif method == 'min':
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(m.view(batch_size, -1))  # min pooling
        elif method == 'avg':
            a = F.avg_pool2d(x, pool_size)
            pool_features.append(a.view(batch_size, -1))  # average pooling
        else:
            m1  = F.avg_pool2d(x, pool_size)
            rm2 = torch.sqrt(F.relu(F.avg_pool2d(torch.pow(x, 2), pool_size) - torch.pow(m1, 2)))
            if method == 'std':
                pool_features.append(rm2.view(batch_size, -1))  # std pooling
            else:
                pool_features.append(torch.cat((m1, rm2), 1).view(batch_size, -1))  # statistical pooling: mean & std
    return torch.cat(pool_features, dim=1)


class IQAModel(nn.Module):
    def __init__(self, arch='resnext101_32x8d', pool='msa', use_bn_end=False, P6=1, P7=1):
        super(IQAModel, self).__init__()
        self.pool = pool
        self.use_bn_end = use_bn_end
        if pool in ['max', 'min', 'avg', 'std','msa']:
            c = 1
        else:
            c = 2
        self.P6 = P6  #
        self.P7 = P7  #
        features = list(models.__dict__[arch](pretrained=True).children())[:-2]
        if arch == 'alexnet':
            in_features = [256, 256]
            self.id1 = 9
            self.id2 = 12
            features = features[0]
        elif arch == 'vgg16':
            in_features = [512, 512]
            self.id1 = 23
            self.id2 = 30
            features = features[0]
        elif 'res' in arch:
            self.id1 = 6
            self.id2 = 7
            if arch == 'resnet18' or arch == 'resnet34':
                in_features = [256, 512]
            else:
                in_features = [1024, 2048]
        else: 
            print('The arch is not implemented!')
        self.features = nn.Sequential(*features)
       
        self.dr6 = nn.Sequential(nn.Linear(in_features[0] * c * sum([p * p for p in range(1, self.P6+1)]), 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 256),
                                 nn.BatchNorm1d(256),                             
                                 nn.Linear(256, 64),
                                 nn.BatchNorm1d(64), nn.ReLU())
        self.dr7 = nn.Sequential(nn.Linear(in_features[1] * c * sum([p * p for p in range(1, self.P7+1)]), 1024),
                                 nn.BatchNorm1d(1024),                               
                                 nn.Linear(1024, 256),
                                 nn.BatchNorm1d(256),                                 
                                 nn.Linear(256, 64),
                                 nn.BatchNorm1d(64), nn.ReLU())
        self.att6 = nn.Sequential(nn.Linear(1024, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 1024),
                                  nn.Sigmoid())
        self.att7 = nn.Sequential(nn.Linear(2048, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 2048),
                                  nn.Sigmoid())

        self.getw5 = nn.Sequential(nn.Conv2d(1,64,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(64,1,kernel_size=3,padding=1),
                                    nn.ReLU()
                                  )

        self.getw6 = nn.Sequential(nn.Conv2d(1,64,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(64,1,kernel_size=3,padding=1),
                                    nn.ReLU()
                                  )        
        
        
        
        if self.use_bn_end:
            self.regr6 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regr7 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regression = nn.Sequential(nn.Linear(64 * 2, 1), nn.BatchNorm1d(1))
        else:
            self.regr6 = nn.Linear(64, 1)
            self.regr7 = nn.Linear(64, 1)
            self.regression = nn.Linear(64 * 2, 1)

    def extract_features(self, x):
        f, pq = [], []
        xmax,_  = torch.max(x,dim=1,keepdim=True)
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.id1:
#                print(x.shape)
                x6 = SPSP(x,xmax, net =self.getw5,P=self.P6, method=self.pool)
                x6_m, x6_s = torch.split(x6,1024,1)
                x6_m = self.att6(x6_m)
                x6 = x6_m*x6_s
                x6 = self.dr6(x6)
                f.append(x6)
                pq.append(self.regr6(x6))
            if ii == self.id2:
#                print(x.shape)
                x7 = SPSP(x,xmax,net =self.getw6,  P=self.P7, method=self.pool)
                x7_m, x7_s = torch.split(x7,2048,1)
                x7_m = self.att7(x7_m)
                x7 = x7_m*x7_s
                x7 = self.dr7(x7)
                f.append(x7)
                pq.append(self.regr7(x7))

        f = torch.cat(f, dim=1)

        return f, pq

    def forward(self, x):
        f, pq = self.extract_features(x)
        s = self.regression(f)
        pq.append(s)

        return pq


def test():

    net = IQAModel()
    net.cuda()
      
    x1 = torch.randn(2, 3,224,224)
    x1 = Variable(x1).cuda()
    
    pq = net.forward(x1)
    print(pq)
  
if __name__== '__main__':
    test()   