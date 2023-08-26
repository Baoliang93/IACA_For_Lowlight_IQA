import sys
import os
import torch
from torch.optim import Adam, SGD, Adadelta, lr_scheduler
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
import ipdb
eps = 1e-8
#f=open("log.txt","w")
#ftmp=sys.stdout
#sys.stdout=f

def linear_mapping(pq, sq, i=0):
    k = []
    b = []
    ones = np.ones_like(pq)
    yp1 = np.concatenate((pq, ones), axis=1).astype('float64')
    h = np.matmul(np.linalg.inv(np.matmul(yp1.transpose(), yp1)), np.matmul(yp1.transpose(), sq.astype('float64')))
    k.append(h[0].item())
    b.append(h[1].item())

    pq = np.matmul(yp1, h)

    return np.reshape(pq, (-1,)), k, b


def run(args):

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IQAModel(arch=args.arch, pool=args.pool, use_bn_end=args.use_bn_end, P6=args.P6, P7=args.P7).to(device)  #
    
    test_dataset = IQADataset(txt_pth=args.txt_pth,img_pth=args.img_pth,ref_pth=args.ref_pth,\
                               isresize=args.isresize,status='testall')	
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,\
                                               shuffle=False, num_workers=8,\
                                               pin_memory=True)
    checkpoint = torch.load(args.trained_model_file+'_last')
    model.load_state_dict(checkpoint['model'])
############################## evaluater ################################################################            
#    model.eval()
#    srocc = SROCC()
#    plcc = PLCC()
#    rmse = RMSE()
#    with torch.no_grad():
#        for batch, img_ptchs in enumerate(test_loader): 
#            im, ref_im, label,name = img_ptchs
#            im, label = im.cuda(), label.cuda()
#            score = model(im)
#            s = score[-1].cpu()
#            label = label.cpu()
#                        
##            print(s,label)
#            srocc.update(label.numpy(), s.numpy())
#            plcc.update(label.numpy(), s.numpy())
#            rmse.update(label.numpy(), s.numpy())
#        
#    srocc_val = srocc.compute()
#    plcc_val = plcc.compute()
#    rmse_val = rmse.compute()
#    srocc_val = round(srocc_val, 4) 
#    plcc_val = round(plcc_val, 4) 
#    rmse_val = round(rmse_val, 4) 
#    print('***Eval  SROCC: '+str(srocc_val)+' PLCC: '+str(plcc_val)+' RMSE: '+str(rmse_val)+'\n')
##    f.write('***Eval SROCC: '+str(srocc_val)+' PLCC: '+str(plcc_val)+' RMSE: '+str(rmse_val)+'\n')


############################## Linear tester ################################################################            
    model.eval()
    pre = []
    mos = []

    with torch.no_grad():
        for batch, img_ptchs in enumerate(test_loader): 
            im, ref_im, label,name = img_ptchs
            im, label = im.cuda(), label.cuda()
            score = model(im)
            s = score[-1].cpu()
            label = label.cpu()
            pre.append(s)
            mos.append(label)


        sq = np.array(mos, dtype=np.float64).flatten()

        pq_before = np.reshape(np.asarray(pre), (-1, 1))
        pq, k, b =  linear_mapping(pq_before, sq, i=0)
        print(k,b)
        SROCC = stats.spearmanr(sq, pq)[0]
        PLCC = stats.pearsonr(sq, pq)[0]
        RMSE = np.sqrt(((sq - pq) ** 2).mean())
        print('***Eval  SROCC: '+str(SROCC)+' PLCC: '+str(PLCC)+' RMSE: '+str(RMSE)+'\n')

      

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

    args.txt_pth = '../../datasets/SQUARE-LOL/name_mos.txt'
    args.img_pth = '../../datasets/SQUARE-LOL/enhancement/'
    args.ref_pth = '../../datasets/SQUARE-LOL/all_normal/'
    
    
   
   
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
