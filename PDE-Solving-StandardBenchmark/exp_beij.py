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
import wandb

parser = argparse.ArgumentParser('Training Transolver')

def set_seed(seed):    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
torch.backends.cudnn.deterministic = True

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=10_000)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_Irregular_Mesh')
parser.add_argument('--n-hidden', type=int, default=128, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=4, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='beij')
parser.add_argument('--data_path', type=str, default='/data/fno')
parser.add_argument('--shuf-seed', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)


args = parser.parse_args()

set_seed(args.seed)

wandb.login(key='d612cda26a5690e196d092756d668fc2aee8525b')
wandb.init(project='transolver', name=args.save_name)
wandb.config.update(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat'
test_path = args.data_path + '/piececonst_r421_N1024_smooth2.mat'
ntrain = args.ntrain
ntest = 200
epochs = args.epochs
eval = args.eval
save_name = args.save_name

def shuffle(x,y, seed=1):
    np.random.seed(seed)
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    return x,y

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main():
    r = args.downsample
    h = int(((421 - 1) / r) + 1)
    s = h
    dx = 1.0 / s

    ### load data
    ndims, res_1d = 1, 168
    ntrain,ntest = 5000, 1000
    def get_beijing(seed=0, normalization=True):
        Ntr, Nte = 5000, 1000
        import pickle
        with open('../../deep_gp_op/datasets/beijing_data.pickle', 'rb') as handle:
            d = pickle.load(handle)
        X, Y = d["x"][:Ntr+Nte], d["y"][:Ntr+Nte]
        X,Y=shuffle(X,Y,seed=args.seed)
        X = torch.tensor(X, dtype=torch.float32); Y = torch.tensor(Y, dtype=torch.float32)
        Xtr, Xte = X[:Ntr], X[Ntr:]
        Ytr, Yte = Y[:Ntr], Y[Ntr:]
        return Xtr, Xte,Ytr,Yte

    x_train,x_test,y_train,y_test = get_beijing()
    y_train, y_test = y_train.squeeze(), y_test.squeeze()
    x_grid = np.linspace(0,1,x_train.shape[1])[:,None]
    x_grid = torch.tensor(x_grid, dtype=torch.float32)

    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)

    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)

    x_normalizer.cuda()
    y_normalizer.cuda()

    pos = x_grid
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    print("Dataloading is over.")

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                              batch_size=args.batch_size, shuffle=False)

    model = get_model(args).Model(space_dim=1,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=5,
                                  out_dim=1,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(args)
    print(model)
    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)
    de_x = TestLoss(size_average=False)
    de_y = TestLoss(size_average=False)

    if eval:
        print("model evaluation")
        print(s, s)
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
        model.eval()
        showcase = 10
        id = 0
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        with torch.no_grad():
            rel_err = 0.0
            with torch.no_grad():
                for x, fx, y in test_loader:
                    id += 1
                    x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                    out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()

                    rel_err += tl

                    if id < showcase:
                        print(id)
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(out[0, :].reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(
                            os.path.join('./results/' + save_name + '/',
                                         "case_" + str(id) + "_pred.pdf"))
                        plt.close()
                        # ============ #
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(y[0, :].reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(
                            os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_gt.pdf"))
                        plt.close()
                        # ============ #
                        plt.figure()
                        plt.axis('off')
                        plt.imshow((y[0, :] - out[0, :]).reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.clim(-0.0005, 0.0005)
                        plt.savefig(
                            os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_error.pdf"))
                        plt.close()
                        # ============ #
                        plt.figure()
                        plt.axis('off')
                        plt.imshow((fx[0, :].unsqueeze(-1)).reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(
                            os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_input.pdf"))
                        plt.close()

            rel_err /= ntest
            print("rel_err:{}".format(rel_err))
            
    else:
        for ep in range(args.epochs):
            model.train()
            train_loss = 0
            reg = 0
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
            reg /= ntrain
            print("Epoch {} Reg : {:.5f} Train loss : {:.5f}".format(ep, reg, train_loss))

            model.eval()
            rel_err = 0.0
            id = 0
            with torch.no_grad():
                for x, fx, y in test_loader:
                    id += 1
                    if id == 2:
                        vis = True
                    else:
                        vis = False
                    x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                    out = model(x, fx=fx).squeeze(-1)
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest
            print("rel_err:{}".format(rel_err))
            wandb.log({"test_loss": rel_err}, step=ep)

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))


if __name__ == "__main__":
    main()
