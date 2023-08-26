import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
#import pytorch_colors as colors
import numpy as np
import torchvision
import pytorch_ssim
from DISTS_pytorch import DISTS
from IQA_pytorch import LPIPSvgg

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        k = k.squeeze(3).squeeze(2)

        return k

			    
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        
#        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        d = torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2)
        d = d.view(d.shape[0],-1)
        d = torch.mean(d,dim=1,keepdim=True)

        return d
        
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2)
        h_tv = h_tv.view(batch_size,-1)
        h_tv = torch.sum(h_tv,dim=1,keepdim=True)
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2)
        w_tv = w_tv.view(batch_size,-1)
        w_tv = torch.sum(w_tv,dim=1,keepdim=True)
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)

class L_sa(nn.Module):
    def __init__(self):
        super(L_sa, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        ba,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        
        k = k.view(ba,-1)
        k = torch.mean(k,dim=1,keepdim=True)
    
        return k

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        bat = input.shape[0]
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
#        loss = 0.0
        x = input
        y = target
        idx = 0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            if idx==0:
               out = torch.nn.functional.l1_loss(x, y,reduction='none').view(bat,-1).mean(1,keepdim=True)
               idx=1
            else:
               diff = torch.nn.functional.l1_loss(x, y,reduction='none').view(bat,-1).mean(1,keepdim=True)
               out = torch.cat((out,diff),1)
               
        return out
    
    
class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = [h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3]
        return out
    

class mse_loss(nn.Module):
    def __init__(self):
        super(mse_loss, self).__init__()
        self.evalator = nn.MSELoss(reduction='none').cuda()   
    
    def forward(self, x,y):
        bat = x.shape[0]
        out = self.evalator(x,y).contiguous().view(bat,-1).mean(1,keepdim=True)
        return out 
        

class ssim_loss(nn.Module):
    def __init__(self):
        super(ssim_loss, self).__init__()
        self.evalator = pytorch_ssim.SSIM(window_size = 11,size_average = False).cuda() 
    
    def forward(self, x,y):
        out = self.evalator(x,y).unsqueeze(1)
        return out 
    
    

class dist_loss(nn.Module):
    def __init__(self):
        super(dist_loss, self).__init__()
        self.evalator = DISTS() 
    
    def forward(self, x,y):
        out = self.evalator(x,y,require_grad=True, batch_average=False)
        if x.shape[0]>1:
            out=out.unsqueeze(1)
        else:
            out=out.unsqueeze(0).unsqueeze(1)
            
        return out     

class lpipsVgg_loss(nn.Module):
    def __init__(self):
        super(lpipsVgg_loss, self).__init__()
        self.evalator = LPIPSvgg()     
    def forward(self, x,y):
        out = self.evalator(x,y,as_loss=False).unsqueeze(1)
        return out      
    
def test():
   
    net = dist_loss() 
    net.cuda()
      
    x1 = torch.randn(1, 3,224,224)
    x1 = x1.cuda()
    
      
    x2 = torch.randn(1, 3,224,224)
    x2 = x2.cuda()
    out = net.forward(x1,x2)
    print(out.shape)
    
if __name__== '__main__':
    test()       
    
    
    
    
    
    