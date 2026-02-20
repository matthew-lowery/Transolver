import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import *
from utils.testloss import TestLoss
from einops import rearrange
from model_dict import get_model
from utils.normalizer import UnitTransformer, UnitTransformerAll
import matplotlib.pyplot as plt
import time
import wandb
from scipy.io import loadmat

parser = argparse.ArgumentParser('Training Transolver')

def set_seed(seed):    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_Irregular_Mesh')
parser.add_argument('--n-hidden', type=int, default=32, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=4, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=500)
parser.add_argument('--ntest', type=int, default=100)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--project-name', type=str, default='transolver_manifold_many')
parser.add_argument('--slice-num', type=int, default=16)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--norm-grid', type=int, default=0) 
parser.add_argument('--dir', type=str, default='/projects/bfel/mlowery/manifold_datasets')
parser.add_argument('--coords', action='store_true')
# parser.add_argument('--dir', type=str, default='../../matlab_master_rbffdcodes/Manifolds/OperatorLearning/manifold_datasets/')
parser.add_argument('--n opoints', default=2400) ### torus: 2400, 5046, 10086; sphere = 2562, 5762, 10242
parser.add_argument('--val', action='store_true')
parser.add_argument('--problem', type=str, default='rd', choices=['rd'])
parser.add_argument('--ktrain', type=int, default=2) ### max is 12
parser.add_argument('--ktest', type=int, default=5)
args = parser.parse_args()
set_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

epochs = args.epochs


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    ########## load data ########################################################################

    ### wants N,n,1 input for funcs
    if not args.wandb:
        os.environ["WANDB_MODE"] = "disabled"
    wandb.login(key='d612cda26a5690e196d092756d668fc2aee8525b')
    wandb.init(project=args.project_name, name=f'{args.problem}_{args.ktrain}')
    wandb.config.update(args)

    data = loadmat(os.path.join(args.dir, f'frac_{args.problem}_isPositive_manifold_many.mat'))
    f_train = np.stack([x[0] for x in data['F_train']])[:args.ktrain, :args.ntrain] # (k, n, n) -> (17, 1200, 2500)
    f_test = np.stack([x[0] for x in data['F_test']])[:args.ktest, :args.ntest]
    u_train = np.stack([x[0] for x in data['U_train']])[:args.ktrain, :args.ntrain]
    u_test = np.stack([x[0] for x in data['U_test']])[:args.ktest, :args.ntest]

    nk, _, n = f_train.shape
    print(f_train.shape, f_test.shape)

    if args.coords:
        x = np.stack([x[0] for x in data['surfs']]) # (k, n, 3)

    else:
        k = data['t'][0]
        x = np.repeat(k[:,None,None], n, axis=1) # (k, n, 1)
        

    x_train = f_train.reshape(f_train.shape[0] * f_train.shape[1], -1, 1)
    x_test = f_test.reshape(f_test.shape[0] * f_test.shape[1], -1, 1)

    y_train = u_train.reshape(u_train.shape[0] * u_train.shape[1], -1)
    y_test = u_test.reshape(u_test.shape[0] * u_test.shape[1], -1)

    x_grid_train = np.repeat(x[:args.ktrain,None], f_train.shape[1], axis=1) # ktr, Ntr, n, 1/3
    x_grid_train = x_grid_train.reshape(len(x_train), n, -1)
    x_grid_test = np.repeat(x[:args.ktest,None], f_test.shape[1], axis=1) # kte, Nte, n, 1/3
    x_grid_test = x_grid_test.reshape(len(x_test), n, -1)

    print(f'{x_train.shape=}, {x_test.shape=}, {y_train.shape=}, {y_test.shape=}')
    print(f'{x_grid_test.shape=}, {x_grid_train.shape=}')

    # range norm
    if args.norm_grid == 2:
        xmi, xma = np.min(x_grid_train, keepdims=True), np.max(x_grid_train, keepdims=True)
        x_grid_train = (x_grid_train - xmi) / ((xma - xmi) + 1e-8)
        x_grid_test = (x_grid_test - xmi) / ((xma - xmi) + 1e-8)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test =  torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    x_grid_train = torch.tensor(x_grid_train, dtype=torch.float32)
    x_grid_test = torch.tensor(x_grid_test, dtype=torch.float32)
    ###########################

    ntrain = len(x_train)
    ntest = len(x_test)

    # mean/std norm
    if args.norm_grid == 3: # 
        x_grid_normalizer = UnitTransformerAll(x_grid_train)
        x_grid_train = x_grid_normalizer.encode(x_grid_train)
        x_grid_test = x_grid_normalizer.encode(x_grid_test)

 
    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)

    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)

    x_normalizer.cuda()
    y_normalizer.cuda()

    pos_train = x_grid_train 
    pos_test = x_grid_test 

    print("Dataloading is over.")
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                              batch_size=args.batch_size, shuffle=False)
    in_channels = x_train.shape[-1]
    out_channels = 1

    model = get_model(args).Model(space_dim=3,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False, ### what this be
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=in_channels,
                                  out_dim=out_channels,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(args)
    print(model)
    num_params = count_parameters(model)
    wandb.log({'param_count': num_params})

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    train_t1 = time.perf_counter()
    for ep in range(args.epochs):
        
        model.train()
        train_loss = 0
        for x, fx, y in train_loader:
            x, fx, y = x.cuda(), fx.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x, fx=fx).squeeze(-1)  # B, N , 2, fx: B, N, y: B, N
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            l2loss = myloss(out, y)
            loss = l2loss
            loss.backward()

            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += l2loss.item()
            scheduler.step()

        train_loss /= ntrain
        print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))
        wandb.log({'train_loss': train_loss}, step=ep)
    train_t2 = time.perf_counter()
       
    rel_err = 0.0
    eval_t1 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for x, fx, y in test_loader:
            x, fx, y = x.cuda(), fx.cuda(), y.cuda()
            out = model(x, fx=fx).squeeze(-1)
            out = y_normalizer.decode(out)
            out = torch.linalg.norm(out, dim=-1)
            y = torch.linalg.norm(y, dim=-1)
            tl = myloss(out, y).item()
            rel_err += tl
    rel_err /= ntest
    print("rel_err:{}".format(rel_err))
    eval_t2 = time.perf_counter()
    print('eval_time: ', eval_t2-eval_t1, 'train_time: ', train_t2-train_t1)
    wandb.log({"test_loss": rel_err, 'eval_time:': eval_t2-eval_t1, 'train_time': train_t2 - train_t1}, step=ep, commit=True)


if __name__ == "__main__":
    main()
