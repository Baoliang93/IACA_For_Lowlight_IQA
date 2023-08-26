import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
#import pytorch_colors as colors
import numpy as np
import torchvision

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()
#        self.pool = nn.AdaptiveAvgPool2d(patch_size)
    def forward(self, x):
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        r,g, b = torch.split(x, 1, dim=1)
        out = torch.pow(r-mg,2)
        Drb = torch.pow(r-mb,2)
        out = torch.cat((out,Drb),1)
        Dgb = torch.pow(b-mg,2)
        out = torch.cat((out,Dgb),1)

        return out

			    
class L_exp(nn.Module):

    def __init__(self,patch_size=16,mean_val=0.6):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        mean = F.interpolate(mean,size=(h,w),mode='nearest')
#        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        d = torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2)

        return d


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)
        return E
        
#class L_TV(nn.Module):
#    def __init__(self):
#        super(L_TV,self).__init__()
#
#    def forward(self,x):
#        b,c,h,w = x.shape
#        h_x = x.size()[2]
#        w_x = x.size()[3]
#        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2)
#        h_tv = F.interpolate(h_tv,size=(h,w),mode='nearest')
#        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2)
#        w_tv = F.interpolate(w_tv,size=(h,w),mode='nearest')
#        all_tv = torch.cat((h_tv,w_tv),1)
#
#        return all_tv


class LRPro(nn.Module):

    def __init__(self,):
        super(LRPro,self).__init__()
        self.L_color = L_color()
        self.L_exp = L_exp()
#        self.L_TV = L_TV()

    def forward(self,x):
        out = self.L_color(x)
        exp =  self.L_exp(x)
#        tv = self.L_TV(x)
        out = torch.cat((out,exp),1)
        return out
    
def test():
   
    net = LRPro() 
    net.cuda()
      
    x1 = torch.randn(2, 3,600,400)
    x1 = x1.cuda()
    
    out = net.forward(x1)
    print(out.shape)
    
if __name__== '__main__':
    test()       
    
    