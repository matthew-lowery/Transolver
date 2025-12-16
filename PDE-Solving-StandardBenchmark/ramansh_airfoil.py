import os
import argparse
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from tqdm import *
from utils.testloss import TestLoss
from einops import rearrange
from model_dict import get_model
from utils.normalizer import UnitTransformer
import matplotlib.pyplot as plt
import time
import wandb
import scipy
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
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--project-name', type=str, default='ramansh_transolver3')
parser.add_argument('--slice-num', type=int, default=16)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--norm-grid', action='store_true')
parser.add_argument('--calc-div', action='store_true')
parser.add_argument('--div-folder', type=str, default='/projects/bfel/mlowery/transolver_divs')
parser.add_argument('--dir', type=str, default='/projects/bfel/mlowery/geo-fno-new')
parser.add_argument('--model-folder', type=str, default='/projects/bfel/mlowery/transolver_models')
parser.add_argument('--dataset', type=str, default='airfoil')

args = parser.parse_args()
set_seed(args.seed)

name = f"{args.dataset}_{args.seed}_{args.ntrain}_all" ## all as in no subsample nymore 
if not args.wandb:
    os.environ["WANDB_MODE"] = "disabled"
wandb.login(key='d612cda26a5690e196d092756d668fc2aee8525b')
wandb.init(project=args.project_name)
wandb.config.update(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

ntrain = args.ntrain
epochs = args.epochs


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

''' ok so here the problem is from 2k 2d points to a 3d function at 500 different points. Transolver can only predict on the input 2k mesh grid, so we just take the first 500 points of the output'''
def main():
    ########## load data ########################################################################
    # data = np.load(os.path.join(args.dir, f'{args.dataset}.npz'))
    data = np.load(f'/home/matt/ram_dataset/geo-fno-new/{args.dataset}.npz')
    print(data.keys())
    x_grid = data['x_grid']; y = data['y']
    print(y.shape)
    y_train = y[:args.ntrain]; y_test = y[-200:]
    x_grid_train = x_grid[:args.ntrain]; x_grid_test = x_grid[-200:]

    #### range normalize input coords to [0,1]^2
    min, max = np.min(x_grid_train, axis=(0,1), keepdims=True), np.max(x_grid_train, axis=(0,1),keepdims=True)
    x_grid_train = (x_grid_train - min) / ((max-min) + 1e-6)
    x_grid_test = (x_grid_test - min) / ((max-min) + 1e-6)

    # pad = np.zeros((*x_grid_train.shape[:2], 1))
    # x_grid_train = np.concatenate((x_grid_train, pad), axis=-1)
    # pad = np.zeros((*x_grid_test.shape[:2], 1))
    # x_grid_test = np.concatenate((x_grid_test, pad), axis=-1)
    # print(x_grid_train.shape, x_grid_test.shape) (1000, 2000, 3) (200, 2000, 3)

    x_grid_train = torch.tensor(x_grid_train, dtype=torch.float32)
    x_grid_test =  torch.tensor(x_grid_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    ###########################
    # print(y_train.shape, y_test.shape) # ([1000, 500, 3]) torch.Size([200, 500, 3])
    # pad = torch.zeros((y_train.shape[0], 1500, y_train.shape[-1]))
    # y_train = torch.cat((y_train, pad), axis=1) # print(y_train.shape) torch.Size([1000, 2000, 3])
    # pad = torch.zeros((y_test.shape[0], 1500, y_test.shape[-1]))
    # y_test = torch.cat((y_test, pad), axis=1)

    y_normalizer = UnitTransformer(y_train)
    y_train = y_normalizer.encode(y_train)
    y_normalizer.cuda()
    print(x_grid_train.shape, y_train.shape, x_grid_test.shape, y_test.shape)
    print("Dataloading is over.")

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_grid_train, y_train),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_grid_test, y_test),
                                              batch_size=args.batch_size, shuffle=False)
    # in_channels = x_train.shape[-1]
    out_channels = y_train.shape[-1]

    model = get_model(args).Model(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False, ### what this be
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=0,
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
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x, fx=None).squeeze(-1)  # B, N , 2, fx: B, N, y: B, N
            out = out[:, :500]
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
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x, fx=None).squeeze(-1)
            out = out[:, :500]
            out = y_normalizer.decode(out)
            tl = myloss(out, y).item()
            rel_err += tl
    rel_err /= ntest
    print("rel_err:{}".format(rel_err))
    eval_t2 = time.perf_counter()
    print('eval_time: ', eval_t2-eval_t1, 'train_time: ', train_t2-train_t1)
    wandb.log({"test_loss": rel_err, 'eval_time:': eval_t2-eval_t1, 'train_time': train_t2 - train_t1}, step=ep, commit=True)
    if args.calc_div:
        y_preds_test = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x, fx=None).squeeze(-1)
                out = out[:, :500]
                out = y_normalizer.decode(out)
                y_preds_test.append(out)
                # tl = myloss(out, y).item()
                rel_err += tl

        y_preds_test = torch.stack(y_preds_test).reshape(ntest, -1, out_channels)

    ### saving model for later use
    if args.save:
        os.makedirs(args.model_folder, exist_ok=True)
        torch.save({
        "model_state_dict": model.state_dict(),
        }, os.path.join(args.model_folder, f'{name}.torch'))

        ### saving test output functions for div calc 
        os.makedirs(args.div_folder, exist_ok=True)
        scipy.io.savemat(os.path.join(args.div_folder, f'{name}.mat'), {'x_grid': data['x_grid'],
                                                                        'x_grid_norm': x.cpu().numpy().astype(np.float64),
                                                                        'y_preds_test': y_preds_test.cpu().numpy().astype(np.float64)})

if __name__ == "__main__":
    main()
