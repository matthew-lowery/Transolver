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
#SBATCH --time=15:00:00
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
#
#for seed in 1 2 3; do
#for nh in 64; do
#sp "python3 exp_ns.py --epochs=500 --slice_num=32 --n-heads=8 --n-hidden=$nh --n-layers=8 --seed=$seed"
#done
#done
#
for seed in 1 2 3; do
sp "python3 exp_beij.py --slice_num=32 --n-heads=8 --n-hidden=256 --n-layers=8 --seed=$seed" 10
done
