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
from utils import AverageMeter, SROCC, PLCC, RMSE
from IQA_Loss import IQALoss

eps = 1e-8
f=open("log.txt","w")
#ftmp=sys.stdout
#sys.stdout=f

def run(args):

    global best_val_criterion, best_epoch
    best_val_criterion, best_epoch = -100, -1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IQAModel(arch=args.arch, pool=args.pool, use_bn_end=args.use_bn_end, P6=args.P6, P7=args.P7).to(device)  #
    print(model)
    if args.ft_lr_ratio == .0:
        for param in model.features.parameters():
            param.requires_grad = False
       
    train_dataset = IQADataset(txt_pth=args.txt_pth,img_pth=args.img_pth,ref_pth=args.ref_pth,\
                               isresize=args.isresize,status='train')	
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,\
                                               shuffle=True, num_workers=8,\
                                               pin_memory=True)
   
    test_dataset = IQADataset(txt_pth=args.txt_pth,img_pth=args.img_pth,ref_pth=args.ref_pth,\
                               isresize=args.isresize,status='testall')	
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,\
                                               shuffle=False, num_workers=8,\
                                               pin_memory=True)
    
    val_dataset = IQADataset(txt_pth=args.txt_pth,img_pth=args.img_pth,ref_pth=args.ref_pth,\
                               isresize=args.isresize,status='val')	
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,\
                                               shuffle=False, num_workers=8,\
                                               pin_memory=True)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)



#################################  trainer ########################################

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    loss_func = IQALoss(loss_type=args.loss_type, alpha=args.alpha, beta=args.beta, p=args.p, q=args.q, 
                        monotonicity_regularization=args.monotonicity_regularization, gamma=args.gamma, detach=args.detach).cuda()
    for epoch in range(args.epochs+1):
        model.train()
        for batch, impt  in enumerate(train_loader):
             im, ref_im, label,name = impt
             x, y = im.cuda(), label.cuda()
             y = y.float()
             y_pred = model(x)
             loss = loss_func(y_pred, y) / args.accumulation_steps
             with amp.scale_loss(loss, optimizer) as scaled_loss:
                  scaled_loss.backward()
             if batch % args.accumulation_steps == 0:
                 optimizer.step()
                 optimizer.zero_grad()
                 print('train_loss:'+str(loss.data.cpu().numpy())+' '+str(batch)+'/'+str( len(train_loader)))
                 f.write('train_loss:'+str(loss.data.cpu().numpy())+' '+str(batch)+'/'+str( len(train_loader))+'\n')
#                 writer.add_scalar("train/loss", loss.data, batch)
        scheduler.step()
############################# tester  ################################################################            
        model.eval()
        srocc = SROCC()
        plcc = PLCC()
        rmse = RMSE()
        with torch.no_grad():
            for batch, img_ptchs in enumerate(test_loader): 
                im, ref_im, label,name = img_ptchs
                im, label = im.cuda(), label.cuda()
                score = model(im)
                s = score[2].cpu()
                label = label.cpu()
                srocc.update(label.numpy(), s.numpy())
                plcc.update(label.numpy(), s.numpy())
                rmse.update(label.numpy(), s.numpy())
            
        srocc_val = srocc.compute()
        plcc_val = plcc.compute()
        rmse_val = rmse.compute()
        srocc_val = round(srocc_val, 4) 
        plcc_val = round(plcc_val, 4) 
        rmse_val = round(rmse_val, 4) 
        print('***Test Epoc'+str(epoch)+' SROCC: '+str(srocc_val)+' PLCC: '+str(plcc_val)+' RMSE: '+str(rmse_val)+'\n')
        f.write('***Test Epoc'+str(epoch)+' SROCC: '+str(srocc_val)+' PLCC: '+str(plcc_val)+' RMSE: '+str(rmse_val)+'\n')

        val_criterion =  srocc_val
        
        
        if val_criterion > best_val_criterion: # If RMSE is used, then change ">" to "<".
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),               
            }
            torch.save(checkpoint, args.trained_model_file+'_best')
            best_val_criterion = val_criterion
            best_epoch = epoch
            print('Save current best model at best_epoch:', best_epoch)
        else:
            print('Best Model is not updated' )

          
############################# 100th epoch ###############################################
    checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),               
            }
    torch.save(checkpoint, args.trained_model_file+'_last')
    
    
if __name__ == "__main__":
    parser = ArgumentParser(description='Norm-in-Norm Loss with Faster Convergence and Better Performance for Image Quality Assessment')
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='batch size for training (default: 8)')
    parser.add_argument('--ft_lr_ratio', type=float, default=0.1,
                        help='ft_lr_ratio (default: 0.1)')
    parser.add_argument('-accum', '--accumulation_steps', type=int, default=1,
                        help='accumulation_steps for training (default: 1)')
    parser.add_argument('-e', '--epochs', type=int, default=100,
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
