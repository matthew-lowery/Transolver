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
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgcs-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=3:00:00
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

for seed in 1 2 3; do
sp "python3 ramansh_taylor_green_time.py --data-root=/projects/bgcs/mlowery/ram_dataset --seed=$seed --ntrain=5000 --norm-grid --wandb --n-heads=6 --n-layers=5 --n-hidden=64 --slice-num=32 --save --calc-div"
sp "python3 ramansh_airfoil.py --ntrain=1000 --seed=$seed --ntrain=5000 --norm-grid --wandb --n-heads=6 --n-layers=5 --n-hidden=128 --slice-num=32 --save --calc-div"
sp "python3 ramansh_species_transport.py --data-root=/projects/bgcs/mlowery/ram_dataset --seed=$seed --ntrain=10000 --norm-grid --wandb --n-heads=4 --n-layers=5 --n-hidden=128 --slice-num=32 --save --calc-div"
sp "python3 ramansh_2d.py --data-root=/projects/bgcs/mlowery/ram_dataset --calc-div --norm-grid --save --seed=$seed --wandb --dataset='flow_cylinder_shedding' --n-heads=4 --n-layers=5 --n-hidden=128 --slice-num=32 --ntrain=10000"
sp "python3 ramansh_2d.py --data-root=/projects/bgcs/mlowery/ram_dataset --calc-div --norm-grid --save --seed=$seed --wandb --dataset='backward_facing_step' --slice-num=16 --n-hidden=128 --n-layers=5 --n-heads=8 --ntrain=500"
sp "python3 ramansh_2d.py --data-root=/projects/bgcs/mlowery/ram_dataset --calc-div --norm-grid --save --seed=$seed --wandb --dataset='backward_facing_step_ood' --slice-num=16 --n-hidden=128 --n-layers=5 --n-heads=8 --ntrain=100"
sp "python3 ramansh_2d.py --data-root=/projects/bgcs/mlowery/ram_dataset --calc-div --norm-grid --save --seed=$seed --wandb --dataset='buoyancy_cavity_flow' --slice-num=32 --n-hidden=128 --n-layers=5 --n-heads=4 --ntrain=10000"
sp "python3 ramansh_2d.py --data-root=/projects/bgcs/mlowery/ram_dataset --calc-div --norm-grid --save --seed=$seed --wandb --dataset='lid_cavity_flow' --slice-num=16 --n-hidden=128 --n-layers=5 --n-heads=4 --ntrain=10000"
sp "python3 ramansh_2d.py --data-root=/projects/bgcs/mlowery/ram_dataset --calc-div --norm-grid --save --seed=$seed --wandb --dataset='taylor_green_exact' --slice-num=64 --n-hidden=128 --n-layers=5 --n-heads=4 --ntrain=5000"
sp "python3 ramansh_2d.py --data-root=/projects/bgcs/mlowery/ram_dataset --calc-div --norm-grid --save --seed=$seed --wandb --dataset='merge_vortices_easier' --slice-num=64 --n-hidden=128 --n-layers=5 --n-heads=8 --ntrain=500"
sp "python3 ramansh_2d.py --data-root=/projects/bgcs/mlowery/ram_dataset --calc-div --norm-grid --save --seed=$seed --wandb --dataset='flow_cylinder_laminar' --slice-num=32 --n-hidden=128 --n-layers=5 --n-heads=8 --ntrain=100"
done
