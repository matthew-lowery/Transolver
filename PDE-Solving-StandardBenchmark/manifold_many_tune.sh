#!/bin/bash


sp() {
    local pycmd="$1"
    local hr="${2:-1}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgcs-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=20:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --output=./k/%x_%A.out
#SBATCH --error=./k/%x_%A.err

module purge
export PATH=/u/mlowery/.conda/envs/gnot/bin:\$PATH
cd /u/mlowery/Transolver/PDE-Solving-StandardBenchmark
$pycmd
EOF
}
# poisson = 4, rd = 0, 

#sp "python3 -u manifold.py --project-name=manifold_final --seed=$seed --n-heads=4 --norm-grid=0 --n-hidden=128 --n-layers=6 --slice-num=32 --epochs=500 --ntrain=$ntrain --problem='nlpoisson' --wandb --surf='torus --npoints=$npoints" 10
#sp "python3 -u manifold.py --project-name=manifold_final --seed=$seed --n-heads=6 --norm-grid=2 --n-hidden=64 --n-layers=6 --slice-num=32 --epochs=500 --ntrain=$ntrain --problem='poisson' --wandb --surf='torus' --npoints=$npoints" 10
#sp "python3 -u manifold.py --project-name=manifold_final --seed=$seed --n-heads=6 --norm-grid=2 --n-hidden=64 --n-layers=6 --slice-num=32 --epochs=500 --ntrain=$ntrain --problem='poisson' --wandb --surf='sphere' --npoints=$npoints" 10
#sp "python3 -u manifold.py --project-name=manifold_final --seed=$seed --n-heads=4 --norm-grid=3 --n-hidden=64 --n-layers=4 --slice-num=16 --epochs=500 --ntrain=$ntrain --problem='nlpoisson' --wandb --surf='sphere' --npoints=$npoints" 10
#sp "python3 -u manifold.py --project-name=manifold_final --seed=$seed --n-heads=4 --norm-grid=3 --n-hidden=128 --n-layers=8 --slice-num=64 --epochs=500 --ntrain=$ntrain --problem='ADRSHEAR' --wandb" 10
#


for ktrain in 2 4 6 8 10 12; do
for seed in 1 2 3; do
sp "python3 -u manifold_many.py --seed=$seed --n-heads=8 --batch-size=20 --norm-grid=3 --n-hidden=64 --n-layers=4 --slice-num=32 --epochs=500 --k-train=$ktrain --problem='rd' --wandb" 
sp "python3 -u manifold_many.py --seed=$seed --n-heads=4 --batch-size=20 --norm-grid=0 --n-hidden=128 --n-layers=4 --slice-num=64 --epochs=500 --k-train=$ktrain --problem='poisson' --wandb" 
done
done

