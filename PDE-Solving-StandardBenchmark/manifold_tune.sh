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
#SBATCH --time=${hr}:00:00
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
 #torus: 2400, 5046, 10086; sphere = 2562, 5762, 10242
# best norm for poisson torus is 2, best norm for nlpoisson torus is 0
#surf='torus'
#for prob in 'nlpoisson'; do
#for npoints in 2400; do
#for ntrain in 1000; do
#for nheads in 4 6 8; do
#for nhidden in 32 64 128; do
#for nlayers in 4 6 8; do
#for slicenum in 16 32 64; do
#sp "python3 -u manifold.py --n-heads=$nheads --norm-grid=0 --n-hidden=$nhidden --n-layers=$nlayers --slice-num=$slicenum --val --epochs=500 --ntrain=$ntrain --problem=$prob --wandb --surf=$surf --npoints=$npoints" 
#done
#done
#done
#done
#done
#done
#done
#
#surf='torus'
#for prob in 'poisson'; do
#for npoints in 2400; do
#for ntrain in 1000; do
#for nheads in 4 6 8; do
#for nhidden in 32 64 128; do
#for nlayers in 4 6 8; do
#for slicenum in 16 32 64; do
#sp "python3 -u manifold.py --n-heads=$nheads --norm-grid=2 --n-hidden=$nhidden --n-layers=$nlayers --slice-num=$slicenum --val --epochs=500 --ntrain=$ntrain --problem=$prob --wandb --surf=$surf --npoints=$npoints" 
#done
#done
#done
#done
#done
#done
#done

surf='sphere'
for prob in 'poisson' 'nlpoisson'; do
for npoints in 2562; do
for ntrain in 1000; do
for nheads in 4; do
for nhidden in 32; do
for nlayers in 4; do
for slicenum in 16; do
for ng in 0 2 3; do
sp "python3 -u manifold.py --n-heads=$nheads --norm-grid=$ng --n-hidden=$nhidden --n-layers=$nlayers --slice-num=$slicenum --val --epochs=500 --ntrain=$ntrain --problem=$prob --wandb --surf=$surf --npoints=$npoints" 
done
done
done
done
done
done
done
done

for prob in 'ADRSHEAR'; do
for npoints in 2562; do
for ntrain in 1000; do
for nheads in 4; do
for nhidden in 32; do
for nlayers in 4; do
for slicenum in 16; do
for ng in 0 2 3; do
sp "python3 -u manifold.py --n-heads=$nheads --norm-grid=$ng --n-hidden=$nhidden --n-layers=$nlayers --slice-num=$slicenum --val --epochs=500 --ntrain=$ntrain --problem=$prob --wandb --surf=$surf --npoints=$npoints" 
done
done
done
done
done
done
done
done

#prob='ADRSHEAR'
#for ntrain in 1000; do
#for k in 'dam2'; do
#sp "python3 -u gp_manifold.py --epochs=1000 --problem=$prob --wandb --ntrain=$ntrain --dir='/projects/bfel/mlowery/manifold_datasets' --kernel=$k"
#done
#done


#for ktr in 12; do
#for prob in 'rd' 'poisson'; do
#for k in 'dam2'; do
#sp "python3 -u gp_manifold_many.py --epochs=1000 --problem=$prob --wandb --kernel=$k --dir='/projects/bfel/mlowery/manifold_datasets' --k-train=$ktr"
#done
#done
#done
#
