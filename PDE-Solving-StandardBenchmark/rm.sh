#!/bin/bash


sp() {
    local pycmd="$1"
    local hr="${2:-8}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --account=bfel-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=8:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --output=./out/%x_%A.out
#SBATCH --error=./err/%x_%A.err

module purge
export PATH=/u/mlowery/.conda/envs/gnot/bin:\$PATH
cd /u/mlowery/Transolver/PDE-Solving-StandardBenchmark/
$pycmd
EOF
}
### Taylor green exact: width=32,heads=4,layer=4, nslices=64
### Merge vortices: width=128,heads=4, layer=5, nslices=32
### Buoyancy CF: width=128, heads=8, layer=5, nslices=16
### BFS: width=128, heads=4, layer=5, nslices=32
### Flow cylinder shedding width=128, heads=4, layer=5, nslices=16
### Flow cylinder laminar width=, heads=4, layer=4, nslices=32
### Merge vortices easier width=128, heads=8, layer=4, nslices=16
### lid cavity flow width=128, heads=4, layer=4, nslices=16

for seed in 1 2 3; do
for ntrain in 100 500 1000; do 
sp "python3 ramansh_2d.py --wandb --calc-div --save --dataset='taylor_green_exact' --slice-num=64 --n-hidden=32 --n-heads=4 --n-layers=4"
#sp "python3 ramansh_2d.py --wandb --calc-div --save --dataset='merge_vortices' --slice-num=32 --n-hidden=128 --n-heads=4 --n-layers=5"
#sp "python3 ramansh_2d.py --wandb --calc-div --save --dataset='buoyancy_cavity_flow' --slice-num=16 --n-hidden=128 --n-heads=8 --n-layers=5"
#sp "python3 ramansh_2d.py --wandb --calc-div --save --dataset='backward_facing_step' --slice-num=32 --n-hidden=128 --n-heads=4 --n-layers=5"
#sp "python3 ramansh_2d.py --wandb --calc-div --save --dataset='flow_cylinder_shedding' --slice-num=16 --n-hidden=128 --n-heads=4 --n-layers=4"
#sp "python3 ramansh_2d.py --wandb --calc-div --save --dataset='flow_cylinder_laminar' --slice-num=32 --n-hidden=128 --n-heads=4 --n-layers=4"
#sp "python3 ramansh_2d.py --wandb --calc-div --save --dataset='merge_vortices_easier' --slice-num=16 --n-hidden=128 --n-heads=8 --n-layers=4"
#sp "python3 ramansh_2d.py --wandb --calc-div --save --dataset='merge_vortices_easier' --slice-num=16 --n-hidden=128 --n-heads=4 --n-layers=4"
#sp "python3 ramansh_2d.py --wandb --calc-div --save --dataset='lid_cavity_flow' --slice-num=16 --n-hidden=128 --n-heads=4 --n-layers=4"
done
done

