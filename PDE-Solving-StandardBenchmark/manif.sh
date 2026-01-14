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
#SBATCH --partition=gpuA100x8,gpuA100x4
#SBATCH --account=bfel-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=${hr}:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --output=./k/%x_%A.out
#SBATCH --error=./k/%x_%A.err

module purge
export PATH=/u/mlowery/.conda/envs/jax_eqx/bin:\$PATH
cd /u/mlowery/Transolver/PDE-Solving-StandardBenchmark/
$pycmd
EOF
}

### tune n-hidden, n-layers, slice-num, n-heads
## k exp w/ nlp nan (maybe bc zero not tiny)
surf='torus'
for prob in 'nlpoisson' 'poisson'; do
for npoints in 2400; do
for ntrain in 3000; do
sp "python3 -u gp_manifold.py --epochs=1000 --ntrain=$ntrain --dir='/projects/bfel/mlowery/manifold_datasets' --problem=$prob --wandb --surf=$surf --npoints=$npoints" 
done
done
done

surf='sphere'
for prob in 'nlpoisson' 'poisson'; do
for npoints in 2562; do
for ntrain in 3000; do
sp "python3 -u gp_manifold.py --epochs=1000 --ntrain=$ntrain --dir='/projects/bfel/mlowery/manifold_datasets' --problem=$prob --wandb --surf=$surf --npoints=$npoints" 
done
done
done


#prob='ADRSHEAR'
#for ntrain in 125 250 500 1000 2000 4000 8000 10000 12500; do
#for k in 'dam2'; do
#sp "python3 -u gp_manifold.py --epochs=1000 --problem=$prob --wandb --ntrain=$ntrain --dir='/projects/bfel/mlowery/manifold_datasets' --kernel=$k"
#done
#done
#
#
#for ktr in 2 3 4 5 6 7 8 9 10 11 12; do
#for prob in 'rd' 'poisson'; do
#for k in 'dam2'; do
#sp "python3 -u gp_manifold_many.py --epochs=1000 --problem=$prob --wandb --kernel=$k --dir='/projects/bfel/mlowery/manifold_datasets' --k-train=$ktr"
#done
#done
#done
#
