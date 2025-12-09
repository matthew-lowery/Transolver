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

for dataset in 'backward_facing_step' 'buoyancy_cavity_flow' 'flow_cylinder_laminar' 'flow_cylinder_shedding' 'lid_cavity_flow' 'merge_vortices' 'taylor_green_exact' 'merge_vortices_easier'; do
for nh in 32 64 128; do
sp "python3 ramansh_2d.py --dataset=$dataset --n-hidden=$nh"
done
done

for dataset in 'backward_facing_step' 'buoyancy_cavity_flow' 'flow_cylinder_laminar' 'flow_cylinder_shedding' 'lid_cavity_flow' 'merge_vortices' 'taylor_green_exact' 'merge_vortices_easier'; do
for nl in 3 5; do
sp "python3 ramansh_2d.py --dataset=$dataset --n-layers=$nl"
done
done

for dataset in 'backward_facing_step' 'buoyancy_cavity_flow' 'flow_cylinder_laminar' 'flow_cylinder_shedding' 'lid_cavity_flow' 'merge_vortices' 'taylor_green_exact' 'merge_vortices_easier'; do
for nh in 8; do
sp "python3 ramansh_2d.py --dataset=$dataset --n-heads=$nh"
done
done

for dataset in 'backward_facing_step' 'buoyancy_cavity_flow' 'flow_cylinder_laminar' 'flow_cylinder_shedding' 'lid_cavity_flow' 'merge_vortices' 'taylor_green_exact' 'merge_vortices_easier'; do
for sn in 32 64; do
sp "python3 ramansh_2d.py --dataset=$dataset --slice-num=$sn"
done
done