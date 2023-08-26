import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import shutil
from abc import ABCMeta, abstractmethod
from threading import Lock
from sys import stdout
import numpy as np
from scipy import stats
from IQA_Loss_MyBox import mse_loss,ssim_loss,VGGPerceptualLoss,lpipsVgg_loss
from IQA_Loss_MyBox import dist_loss,L_TV,L_color,L_exp,L_sa


class MMD_loss(nn.Module):
	def __init__(self, kernel_mul = 2.0, kernel_num = 5):
		super(MMD_loss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		return
	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0])+int(target.size()[0])
		total = torch.cat([source, target], dim=0)

		total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		L2_distance = ((total0-total1)**2).sum(2) 
		if fix_sigma:
		    bandwidth = fix_sigma
		else:
		    bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)

	def forward(self, source, target):
		batch_size = int(source.size()[0])
		kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
		XX = kernels[:batch_size, :batch_size]
		YY = kernels[batch_size:, batch_size:]
		XY = kernels[:batch_size, batch_size:]
		YX = kernels[batch_size:, :batch_size]
		loss = torch.mean(XX + YY - XY -YX)
		return loss
    
    
def get_loss_box():
    pmse_loss = mse_loss().cuda()
    pssim_loss = ssim_loss().cuda()
    
    LPIPSvgg_loss = lpipsVgg_loss().cuda()
    DISTS_loss = dist_loss().cuda()
    percep_loss = VGGPerceptualLoss().cuda()
    
    TV_loss = L_TV().cuda()
    color_loss = L_color().cuda()
    exp_loss = L_exp(16,0.6).cuda() 
    sa_loss = L_sa().cuda()

    
    loss_box = []
    mse_box={'model':pmse_loss,'inputnum':2,'outnum':1,'name':'mse_loss'}
    loss_box.append(mse_box)
    
    ssim_box={'model':pssim_loss,'inputnum':2,'outnum':1,'name':'ssim_loss'}
    loss_box.append(ssim_box)
    
    percep_box={'model':percep_loss,'inputnum':2,'outnum':4,'name':'percep_loss'}
    loss_box.append(percep_box)
    
    LPIPSvgg_box={'model':LPIPSvgg_loss,'inputnum':2,'outnum':1,'name':'LPIPSvgg_loss'}
    loss_box.append(LPIPSvgg_box)
    
    DISTS_box={'model':DISTS_loss,'inputnum':2,'outnum':1,'name':'DISTS_loss'}
    loss_box.append(DISTS_box)
    
   
    TV_box={'model':TV_loss,'inputnum':1,'outnum':1,'name':'TV_loss'}
    loss_box.append(TV_box)
    
    color_box={'model':color_loss,'inputnum':1,'outnum':1,'name':'color_loss'}
    loss_box.append(color_box)
    
    exp_box={'model':exp_loss,'inputnum':1,'outnum':1,'name':'exp_loss'}
    loss_box.append(exp_box)
    
    sa_box={'model':sa_loss,'inputnum':1,'outnum':1,'name':'sa_loss'}
    loss_box.append(sa_box)
 
    return loss_box
        
        

class AverageMeter:
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


"""
Metrics for IQA performance
-----------------------------------------

Including classes:
    * Metric (base)
    * MAE
    * SROCC
    * PLCC
    * RMSE

"""

class Metric(metaclass=ABCMeta):
    def __init__(self):
        super(Metric, self).__init__()
        self.reset()
    
    def reset(self):
        self.x1 = []
        self.x2 = []

    @abstractmethod
    def _compute(self, x1, x2):
        return

    def compute(self):
        x1_array = np.array(self.x1, dtype=np.float)
        x2_array = np.array(self.x2, dtype=np.float)
        return self._compute(x1_array.ravel(), x2_array.ravel())

    def _check_type(self, x):
        return isinstance(x, (float, int, np.ndarray))

    def update(self, x1, x2):
        if self._check_type(x1) and self._check_type(x2):
            self.x1.append(x1)
            self.x2.append(x2)
        else:
            raise TypeError('Data types not supported')

class MAE(Metric):
    def __init__(self):
        super(MAE, self).__init__()

    def _compute(self, x1, x2):
        return np.sum(np.abs(x2-x1))

class SROCC(Metric):
    def __init__(self):
        super(SROCC, self).__init__()
    
    def _compute(self, x1, x2):
        return stats.spearmanr(x1, x2)[0]

class PLCC(Metric):
    def __init__(self):
        super(PLCC, self).__init__()

    def _compute(self, x1, x2):
        return stats.pearsonr(x1, x2)[0]

class RMSE(Metric):
    def __init__(self):
        super(RMSE, self).__init__()

    def _compute(self, x1, x2):
        return np.sqrt(((x2 - x1) ** 2).mean())


def limited_instances(n):
    def decorator(cls):
        _instances = [None]*n
        _lock = Lock()
        def wrapper(idx, *args, **kwargs):
            nonlocal _instances
            with _lock:
                if idx < n:
                    if _instances[idx] is None: _instances[idx] = cls(*args, **kwargs)   
                else:
                    raise KeyError('index exceeds maximum number of instances')
                return _instances[idx]
        return wrapper
    return decorator


class SimpleProgressBar:
    def __init__(self, total_len, pat='#', show_step=False, print_freq=1):
        self.len = total_len
        self.pat = pat
        self.show_step = show_step
        self.print_freq = print_freq
        self.out_stream = stdout

    def show(self, cur, desc):
        bar_len, _ = shutil.get_terminal_size()
        # The tab between desc and the progress bar should be counted.
        # And the '|'s on both ends be counted, too
        bar_len = bar_len - self.len_with_tabs(desc+'\t') - 2
        bar_len = int(bar_len*0.8)
        cur_pos = int(((cur+1)/self.len)*bar_len)
        cur_bar = '|'+self.pat*cur_pos+' '*(bar_len-cur_pos)+'|'

        disp_str = "{0}\t{1}".format(desc, cur_bar)

        # Clean
        self.write('\033[K')

        if self.show_step and (cur % self.print_freq) == 0:
            self.write(disp_str, new_line=True)
            return

        if (cur+1) < self.len:
            self.write(disp_str)
        else:
            self.write(disp_str, new_line=True)

        self.out_stream.flush()

    @staticmethod
    def len_with_tabs(s):
        return len(s.expandtabs())

    def write(self, content, new_line=False):
        end = '\n' if new_line else '\r'
        self.out_stream.write(content+end)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output
