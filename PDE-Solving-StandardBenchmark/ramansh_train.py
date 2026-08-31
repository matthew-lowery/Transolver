"""Transolver training entry point for the RAM operator-dataset format."""
import argparse, os, time
import numpy as np
import scipy.io
import torch
import wandb
from scipy.spatial import cKDTree
from scipy.linalg import lstsq
from itertools import product
from model_dict import get_model
from utils.testloss import TestLoss
from utils.normalizer import UnitTransformer
from ram_dataset_loader import load_dataset, load_ood_dataset
from divergence_metrics import summarize_divergence, interior_mask as build_interior_mask

def build_rbf_fd_gradient(points, order=5):
    points=np.asarray(points,float); d=points.shape[1]
    powers=np.asarray([p for p in product(range(order+1),repeat=d) if sum(p)<=order])
    q=len(powers); k=2*q+1; tree=cKDTree(points); rows=np.repeat(np.arange(len(points)),k)
    cols=np.empty_like(rows); weights=np.empty((d,len(rows))); power=min(max(order-(order%2==0),5),11)
    for i, center in enumerate(points):
        dist, idx=tree.query(center,k=k); scale=dist[-1]
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            raise ValueError('RBF-FD stencil contains coincident points')
        local=(points[idx]-center)/scale
        pair=np.linalg.norm(local[:,None]-local[None,:],axis=-1)
        poly=np.prod(local[:,None]**powers[None,:],axis=-1)
        system=np.block([[pair**power,poly],[poly.T,np.zeros((q,q))]])
        rhs=np.zeros((k+q,d)); rhs[:k]=-local*power*(pair[0,:,None]+np.finfo(float).eps)**(power-2)/scale
        for axis in range(d):
            unit=np.zeros(d,int); unit[axis]=1
            rhs[k+np.flatnonzero(np.all(powers==unit,axis=1))[0],axis]=1/scale
        sol=lstsq(system,rhs,lapack_driver='gelsy',check_finite=False)[0]
        sl=slice(i*k,(i+1)*k); cols[sl]=idx; weights[:,sl]=sol[:k].T
    ij=torch.tensor(np.stack((rows,cols)),dtype=torch.long)
    return tuple(torch.sparse_coo_tensor(ij,torch.tensor(w,dtype=torch.float32),(len(points),len(points))).coalesce() for w in weights)

def divergence_loss(u, ops, mask, steps=1):
    b,_,c=u.shape; u=u.reshape(b,-1,steps,c).permute(0,2,1,3).reshape(b*steps,-1,c)
    div=sum(torch.sparse.mm(op,u[...,a].T).T for a,op in enumerate(ops))[:,mask]
    return div.square().mean(dim=1).sum()/steps

def union_grid(a,b):
    allp=np.concatenate((a,b),axis=0); grid=np.unique(allp,axis=0)
    def inds(p):
        return cKDTree(grid).query(p,k=1)[1]
    return grid, inds(a), inds(b)

parser=argparse.ArgumentParser()
parser.add_argument('--dataset',default='flow_cylinder_laminar')
parser.add_argument('--model',default='Transolver_Irregular_Mesh')
parser.add_argument('--data-root',default='/projects/bgcs/mlowery/ram_dataset')
parser.add_argument('--ntrain',type=int,required=True); parser.add_argument('--npoints',type=int,default=0)
parser.add_argument('--epochs',type=int,default=500); parser.add_argument('--batch-size',type=int,default=20)
parser.add_argument('--lr',type=float,default=1e-3); parser.add_argument('--weight_decay',type=float,default=1e-5)
parser.add_argument('--n-hidden',type=int,default=32); parser.add_argument('--n-layers',type=int,default=4); parser.add_argument('--n-heads',type=int,default=4)
parser.add_argument('--unified_pos',type=int,default=0)
parser.add_argument('--slice-num',type=int,default=16); parser.add_argument('--ref',type=int,default=8); parser.add_argument('--mlp_ratio',type=int,default=1); parser.add_argument('--dropout',type=float,default=0.)
parser.add_argument('--gpu',default='0'); parser.add_argument('--seed',type=int,default=1); parser.add_argument('--project-name',default='ramansh_transolver')
parser.add_argument('--wandb',action='store_true'); parser.add_argument('--save',action='store_true'); parser.add_argument('--norm-grid',action='store_true'); parser.add_argument('--div-loss',action='store_true'); parser.add_argument('--calc-div',action='store_true'); parser.add_argument('--div-loss-weight',type=float,default=1.)
parser.add_argument('--div-folder',default='/projects/bfel/mlowery/transolver_divs'); parser.add_argument('--model-folder',default='/projects/bfel/mlowery/transolver_models'); parser.add_argument('--no-ood',action='store_true')
args=parser.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=args.gpu; torch.manual_seed(args.seed); np.random.seed(args.seed)
if not args.wandb: os.environ['WANDB_MODE']='disabled'
wandb.login(key='d612cda26a5690e196d092756d668fc2aee8525b'); wandb.init(project=args.project_name); wandb.config.update(vars(args))

def main():
    ds=load_dataset(args.dataset,args.ntrain,None if args.npoints<=0 else args.npoints,args.data_root)
    grid, in_idx, out_idx=union_grid(ds.input_points,ds.output_points); dim=grid.shape[1]
    def pad(x):
        z=np.zeros((len(x),len(grid),x.shape[-1]),np.float32); z[:,in_idx]=x; return z
    xtr,xts=pad(ds.train_input),pad(ds.test_input); ytr,yts=ds.train_output,ds.test_output
    mins=grid.min(0,keepdims=True); span=np.maximum(grid.max(0,keepdims=True)-mins,1e-12)
    pos=(grid-mins)/span if args.norm_grid else grid
    xtr,xts,ytr,yts=map(lambda z:torch.tensor(z,dtype=torch.float32),(xtr,xts,ytr,yts)); pos=torch.tensor(pos,dtype=torch.float32)
    xn,yn=UnitTransformer(xtr),UnitTransformer(ytr); xtr,xts=xn.encode(xtr),xn.encode(xts); ytr=yn.encode(ytr); xn.cuda(); yn.cuda()
    train_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos.repeat(args.ntrain,1,1),xtr,ytr),batch_size=args.batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos.repeat(len(xts),1,1),xts,yts),batch_size=args.batch_size)
    ops=mask=None; div_steps=1
    if args.div_loss or args.calc_div:
        physical=ds.output_points
        if args.dataset in {'taylor_green_spacetime','taylor_green_time','taylor_green_spacetime_coeffs','taylor_green_time_coeffs'}:
            physical=physical.reshape(-1,4,3)[:,0,:2]; div_steps=4
        ops=tuple(o.cuda() for o in build_rbf_fd_gradient(physical)); mask=torch.ones(len(physical),dtype=torch.bool).cuda()
        metric_mask=build_interior_mask(physical).cuda()
    else:
        metric_mask=None
    model=get_model(args).Model(space_dim=dim,n_layers=args.n_layers,n_hidden=args.n_hidden,dropout=args.dropout,n_head=args.n_heads,Time_Input=False,mlp_ratio=args.mlp_ratio,fun_dim=xtr.shape[-1],out_dim=ytr.shape[-1],slice_num=args.slice_num,ref=args.ref).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=args.lr,epochs=args.epochs,steps_per_epoch=len(train_loader)); loss_fn=TestLoss(size_average=False)
    for ep in range(args.epochs):
        model.train(); total=0
        for p,fx,y in train_loader:
            p,fx,y=p.cuda(),fx.cuda(),y.cuda(); opt.zero_grad(); out=yn.decode(model(p,fx=fx).squeeze(-1))[:,out_idx]; target=yn.decode(y)
            dl=loss_fn(out,target); vl=divergence_loss(out,ops,mask,div_steps) if ops else out.new_zeros(()); (dl+args.div_loss_weight*vl).backward(); opt.step(); sched.step(); total+=dl.item()
        wandb.log({'train_data_loss':total/args.ntrain},step=ep)
    model.eval(); pred=[]; rel=0
    with torch.no_grad():
        for p,fx,y in test_loader:
            out=yn.decode(model(p.cuda(),fx=fx.cuda()).squeeze(-1))[:,out_idx]; pred.append(out.cpu()); rel+=loss_fn(torch.linalg.norm(out,dim=-1),torch.linalg.norm(y.cuda(),dim=-1)).item()
    rel/=len(xts); wandb.log({'test_loss':rel},commit=not args.calc_div)
    if args.calc_div:
        metrics=summarize_divergence(torch.cat(pred).cuda(), ops, metric_mask, div_steps)
        wandb.log(metrics, commit=True)
    if not args.no_ood:
        try:
            _, _, ood_inputs, ood_targets = load_ood_dataset(
                args.dataset, args.npoints or None, args.data_root
            )
            device = next(model.parameters()).device
            ood_inputs = torch.tensor(
                pad(ood_inputs), dtype=torch.float32, device=device
            )
            ood_targets = torch.tensor(
                ood_targets, dtype=torch.float32, device=device
            )
            ood_positions = pos.repeat(len(ood_inputs), 1, 1).to(device)
            with torch.no_grad():
                ood_predictions = yn.decode(model(
                    ood_positions, fx=xn.encode(ood_inputs)
                ).squeeze(-1))[:, out_idx]
            wandb.log({'ood_loss': loss_fn(
                torch.linalg.norm(ood_predictions, dim=-1),
                torch.linalg.norm(ood_targets, dim=-1),
            ).item()})
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print('OOD unavailable:', exc)
    if args.save:
        os.makedirs(args.model_folder,exist_ok=True); torch.save({'model_state_dict':model.state_dict()},os.path.join(args.model_folder,f'{args.dataset}_{args.seed}_{args.ntrain}.torch'))
        if pred:
            os.makedirs(args.div_folder,exist_ok=True); scipy.io.savemat(os.path.join(args.div_folder,f'{args.dataset}_{args.seed}_{args.ntrain}.mat'),{'x_grid':ds.output_points,'y_preds_test':torch.cat(pred).numpy()})
if __name__=='__main__': main()
