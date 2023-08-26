from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.functional import resize, rotate, crop, hflip, to_tensor, normalize
from PIL import Image
import h5py
import os
import numpy as np
import random
import matplotlib.pyplot as plt  
import torch
from argparse import ArgumentParser
import torchvision
from PIL import Image
import torch

def default_loader(path):
    return Image.open(path).convert('RGB')  #

class IQADataset(Dataset):
    def __init__(self, txt_pth, img_pth, ref_pth, TrainNum=2470, tar_size=(224,224),isresize=True, augment=True, hflip_p=0.5, status='train'):
        self.status = status
        self.augment = augment
        self.size_h = tar_size[0]
        self.size_w = tar_size[1]
        self.hflip_p = hflip_p
        self.img_pth = img_pth
        self.ref_pth = ref_pth

        t_fh = open(txt_pth,"r")                     
        t_imgs = []
        t_labels = []

        
        print('Reading...')            
        for line in t_fh:               
            line = line.rstrip()     
            words = line.split()   
            t_labels.append(float(words[1]))
            t_imgs.append(words[0])

                
        self.img_len = len(t_imgs)
        self.ims = []
        self.ref_imgs = []
        
        if status == 'train':
            self.pt_imgs = t_imgs[0:TrainNum] 
            self.labels = t_labels[0:TrainNum] 
        elif status == 'test' or  status == 'testall':
            self.pt_imgs = t_imgs[TrainNum:-1] 
            self.labels = t_labels[TrainNum:-1] 
        elif status == 'val':
            self.pt_imgs = t_imgs[TrainNum-200:TrainNum] 
            self.labels = t_labels[TrainNum-200:TrainNum] 
        else:
            self.pt_imgs = t_imgs 
            self.labels = t_labels
                
            
#        for im_name in self.pt_imgs:
#            im = Image.open(os.path.join(img_pth, im_name))
#            if isresize:
#               im = im.resize((self.size_h,self.size_w), Image.ANTIALIAS)
#            im = (np.asarray(im)/255.0) 
#            im = torch.from_numpy(im).float()
#            im = im.permute(2,0,1)
#                       
#            ref_im_pth = im_name.split('_')[0].replace('low','normal')+'.png'
#            ref_im = Image.open(os.path.join(ref_pth, ref_im_pth))
#            if isresize:
#               ref_im = ref_im.resize((self.size_h,self.size_w), Image.ANTIALIAS)
#            ref_im = (np.asarray(ref_im)/255.0) 
#            ref_im = torch.from_numpy(ref_im).float()
#            ref_im = ref_im.permute(2,0,1)
#            
#         
#            self.ims.append(im)   
#            self.ref_imgs.append(ref_im)
        print('Readed.')
        print('status: ',status, 'length: ', len(self.labels))

        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.img_pth , self.pt_imgs[idx])).convert('RGB')
        if self.status == 'demo':
            ref_im = im
        else:
            ref_im_pth =  self.pt_imgs[idx].split('_')[0].replace('low','normal')+'.png'
            ref_im = Image.open(os.path.join(self.ref_pth , ref_im_pth)).convert('RGB')
        label = self.labels[idx]
        name = self.pt_imgs[idx]
        
        im = self.transform(im,self.status)
        ref_im = self.transform(ref_im,self.status)
        
        return im, ref_im, label,name

    def transform(self, im, status, angle=2, crop_size_h=224, crop_size_w=224, hflip_p=0.5):
        if status == 'train':  # data augmentation
            angle = random.uniform(-angle, angle)
            p = random.random()
            w, h = im.size
            i = random.randint(0, h - crop_size_h)
            j = random.randint(0, w - crop_size_w)

            im = rotate(im, angle)
            if p < hflip_p:
                im = hflip(im)
            im = crop(im, i, j, crop_size_h,crop_size_w)
        
        if status == 'test':
            w, h = im.size
            i = random.randint(0, max(h - crop_size_h,0))
            j = random.randint(0, max(w - crop_size_w,0))
            im = crop(im, i, j, crop_size_h, crop_size_w)
 
        if status == 'testall':
            w, h = im.size
   
           
        im = to_tensor(im)
#        im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        return im
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    


if __name__=='__main__':

  
    txt_pth = './name_mos.txt'
    img_pth = '../datasets/LOL_IQA/enhancement/'
    ref_pth = '../datasets/CVPR-2020-Semi-Low-Light-Dataset/stage1/all_normal/'
    
    train_dataset = IQADataset(txt_pth=txt_pth,img_pth=img_pth,ref_pth=ref_pth,isresize=False,status='test')
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True)  

    for ii, img_ptchs in enumerate(train_loader):
        if ii>12:
            break
        else:
          img = img_ptchs[0]
          ref_img = img_ptchs[1]

        
          
          print('labels: ', img_ptchs[2])
          print('name: ', img_ptchs[3])
 
          concatenated = torch.cat((img,ref_img),0)
          imshow(torchvision.utils.make_grid(concatenated))

      
      
