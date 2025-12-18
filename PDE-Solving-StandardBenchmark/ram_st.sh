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
#SBATCH --account=bfel-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=1:00:00
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

for ns in 32 64; do
#sp "python3 ramansh_taylor_green_time_coeffs.py --norm-grid --wandb --slice-num=$ns"
#sp "python3 ramansh_taylor_green_coeffs.py --norm-grid --wandb --slice-num=$ns"
sp "python3 ramansh_species_transport.py --ntrain=1000 --norm-grid --wandb --slice-num=$ns"
done
#
#for nl in 3 5; do
##sp "python3 ramansh_taylor_green_time_coeffs.py --norm-grid --wandb --n-layers=$nl"
##sp "python3 ramansh_taylor_green_coeffs.py --norm-grid --wandb --n-layers=$nl"
#sp "python3 ramansh_species_transport.py --ntrain=1000 --norm-grid --wandb --n-layers=$nl"
#done
#
#for nh in 64 128; do
##sp "python3 ramansh_taylor_green_time_coeffs.py --norm-grid --wandb --n-hidden=$nh"
##sp "python3 ramansh_taylor_green_coeffs.py --norm-grid --wandb --n-hidden=$nh"
#sp "python3 ramansh_species_transport.py --ntrain=1000 --norm-grid --wandb --n-hidden=$nh"
#done
#
for nh in 4; do
#sp "python3 ramansh_taylor_green_time_coeffs.py --norm-grid --wandb --n-heads=$nh"
#sp "python3 ramansh_taylor_green_coeffs.py --norm-grid --wandb --n-heads=$nh"
sp "python3 ramansh_species_transport.py --ntrain=1000 --norm-grid --wandb --n-heads=$nh"
done


###
#for seed in 1 2 3; do
#sp "python3 ramansh_species_transport.py --ntrain=10000 --norm-grid --wandb --n-heads=4 --n-layers=5 --n-hidden=128 --slice-num=32 --save --calc-div"
#done
