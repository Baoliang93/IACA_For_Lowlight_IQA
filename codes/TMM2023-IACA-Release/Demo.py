import sys
import os
import torch
from torch.optim import Adam, SGD, Adadelta, lr_scheduler
from apex import amp
from IQA_Dataset import IQADataset
from IQA_Model import IQAModel
from tensorboardX import SummaryWriter
import datetime
import numpy as np
import random
from argparse import ArgumentParser
import torch.nn.functional as F
#from utils import AverageMeter, SROCC, PLCC, RMSE
from IQA_Loss import IQALoss
from scipy import stats
eps = 1e-8
from scipy.optimize import curve_fit
from PIL import Image
from torchvision.transforms.functional import resize, rotate, crop, hflip, to_tensor, normalize
import math

def run(args):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IQAModel(arch=args.arch, pool=args.pool, use_bn_end=args.use_bn_end, P6=args.P6, P7=args.P7).to(device)  #
    checkpoint = torch.load(args.trained_model_file+'_last')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    im = Image.open('./low00013_MR_0.png').convert('RGB')
    im = to_tensor(im).unsqueeze(0).cuda()
    score = model(im)
    s = score[-1].cpu()*0.09627324560431309+0.5154855778833177
    print('Pred Quality Score: ', s[0,0].data.cpu().numpy())
    
    im = Image.open('./low00011_RetinexNet_0.png').convert('RGB')
    im = to_tensor(im).unsqueeze(0).cuda()
    score = model(im)
    s = score[-1].cpu()*0.09627324560431309+0.5154855778833177
    print('Pred Quality Score: ', s[0,0].data.cpu().numpy())
    
    
    
         

if __name__ == "__main__":
    parser = ArgumentParser(description='Norm-in-Norm Loss with Faster Convergence and Better Performance for Image Quality Assessment')
    parser.add_argument("--seed", type=int, default=1113)

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('-bs', '--batch_size', type=int, default=8,
                        help='batch size for training (default: 8)')
    parser.add_argument('--ft_lr_ratio', type=float, default=0.1,
                        help='ft_lr_ratio (default: 0.1)')
    parser.add_argument('-accum', '--accumulation_steps', type=int, default=1,
                        help='accumulation_steps for training (default: 1)')
    parser.add_argument('-e', '--epochs', type=int, default=60,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='lr decay (default: 0.1)')
    parser.add_argument('--overall_lr_decay', type=float, default=0.01,
                        help='overall lr decay (default: 0.01)')
    parser.add_argument('--opt_level', default='O1', type=str,
                        help='opt_level for amp (default: O1)')
    parser.add_argument('--randomness', action='store_true',
                        help='Allow randomness during training?')
    
    
    parser.add_argument('--alpha', nargs=2, type=float, default=[1, 0],
                        help='loss coefficient alpha in total loss (default: [1, 0])')
    parser.add_argument('--beta', nargs=3, type=float, default=[.1, .1, 1],
                        help='loss coefficients for level 6, 7, and 6+7 (default: [.1, .1, 1])')
    parser.add_argument('--arch', default='resnext101_32x8d', type=str,
                        help='arch name (default: resnext101_32x8d)')
    parser.add_argument('--pool', default='msa', type=str,
                        help='pool method (default: avg)')
    parser.add_argument('--use_bn_end', action='store_true',
                        help='Use bn at the end of the output?')
   
    parser.add_argument('--P6', type=int, default=1,
                        help='P6 (default: 1)')
    parser.add_argument('--P7', type=int, default=1,
                        help='P7 (default: 1)')
    parser.add_argument('--loss_type', default='Lp', type=str,
                        help='loss type (default: Lp)')
    parser.add_argument('--p', type=float, default=1,
                        help='p (default: 1)')
    parser.add_argument('--q', type=float, default=2,
                        help='q (default: 2)')
    parser.add_argument('--detach', action='store_true',
                        help='Detach in loss?')
    parser.add_argument('--monotonicity_regularization', action='store_true',
                        help='use monotonicity_regularization?')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='coefficient of monotonicity regularization (default: 0.1)')
    parser.add_argument('--isresize', action='store_true',
                        help='Resize?')
    parser.add_argument('-rs_h', '--resize_size_h', default=300, type=int,
                        help='resize_size_h (default: 498)')
    parser.add_argument('-rs_w', '--resize_size_w', default=200, type=int,
                        help='resize_size_w (default: 664)')


    parser.add_argument("--log_dir", type=str, default="runs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--pbar', action='store_true',
                        help='Use progressbar for the training')

    args = parser.parse_args()
    
    if args.lr_decay == 1 or args.epochs < 3:  # no lr decay
        args.lr_decay_step = args.epochs
    else:  # 
        args.lr_decay_step = int(args.epochs/(1+np.log(args.overall_lr_decay)/np.log(args.lr_decay)))

    if not args.randomness:
        torch.manual_seed(args.seed)  #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    args.format_str = '{}'.format(args.arch)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    args.trained_model_file = 'checkpoints/' + args.format_str
    if not os.path.exists('results'):
        os.makedirs('results')
    args.save_result_file = 'results/' + args.format_str
    print(args)
    run(args)
