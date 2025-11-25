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

parser = argparse.ArgumentParser('Training Transolver')


def set_seed(seed):    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=10_000)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_Structured_Mesh_3D')
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
parser.add_argument('--save_name', type=str, default='burg')
parser.add_argument('--data_path', type=str, default='/data/fno')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()
seed_set(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat'
test_path = args.data_path + '/piececonst_r421_N1024_smooth2.mat'
ntrain = 90
ntest = 10
epochs = args.epochs
eval = args.eval
save_name = args.save_name


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

    fp = '../../deep_gp_op/datasets/ns_pdebench_pres_t0_t1_mach_1.0.npz'
    data = np.load(fp)
    dataset = fp.split('/')[-1].split('.')[0]
    x, x_grid, y= data["x"], data["x_grid"], data["y"]
    
    x = x[:,::2,::2,::2].reshape(ntrain + ntest,-1,1)
    y = y[:,::2,::2,::2].reshape(ntrain + ntest,-1)

    x_grid = data['x_grid']
    x_grid = np.asarray(np.meshgrid(x_grid, x_grid, x_grid)).transpose(1,2,3,0)
    x_grid = x_grid[::2,::2,::2].reshape(-1,3)

    x = torch.tensor(x, dtype=torch.float32); x_grid = torch.tensor(x_grid, dtype=torch.float32); y = torch.tensor(y, dtype=torch.float32)
    print(x.shape)

    x_train, x_test = x[:ntrain], x[-ntest:]
    y_train, y_test = y[:ntrain], y[-ntest:]

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

    model = get_model(args).Model(space_dim=3,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=1,
                                  out_dim=1,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  H=64, W=64, D=64).cuda()

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
