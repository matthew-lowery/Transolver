#!/bin/bash

set -euo pipefail

DIV_LOSS_WEIGHT=1.0

SEEDS=(1 2 3)
DATA_DIR='/projects/bfel/mlowery/geo-fno-new'
RUN_DIR="/projects/bfel/mlowery/transolver-div-loss/lambda-$DIV_LOSS_WEIGHT"
DIV_DIR="$RUN_DIR/divs"
MODEL_DIR="$RUN_DIR/models"
PROJECT_NAME='transolver_div_loss'
SBATCH_ACCOUNT='bgcs-delta-gpu'

COMMON_ARGS="--project-name=$PROJECT_NAME --div-folder=$DIV_DIR --model-folder=$MODEL_DIR --dir=$DATA_DIR --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$DIV_LOSS_WEIGHT"

sp() {
    local pycmd="$1"
    local hours="$2"
    local job_name="$3"

    sbatch <<EOF
#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --account=$SBATCH_ACCOUNT
#SBATCH --job-name=$job_name
#SBATCH --time=${hours}:00:00
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

submit_problem() {
    local script="$1" dataset="$2" ntrain="$3" hidden="$4"
    local layers="$5" heads="$6" slices="$7" hours="$8"
    for seed in "${SEEDS[@]}"; do
        sp "python3 $script $COMMON_ARGS --seed=$seed --dataset=$dataset --ntrain=$ntrain --n-hidden=$hidden --n-layers=$layers --n-heads=$heads --slice-num=$slices" "$hours" "div_$dataset"
    done
}

# script, dataset, ntrain, hidden, layers, heads, slices, hours
submit_problem ramansh_2d.py flow_cylinder_laminar 100 128 5 8 32 3
submit_problem ramansh_2d.py flow_cylinder_shedding 10000 128 5 4 32 3
submit_problem ramansh_2d.py lid_cavity_flow 10000 128 5 4 16 3
submit_problem ramansh_2d.py backward_facing_step 500 128 5 8 16 3
submit_problem ramansh_2d.py buoyancy_cavity_flow 10000 128 5 4 32 3
submit_problem ramansh_2d.py taylor_green_exact 5000 128 5 4 64 3
submit_problem ramansh_taylor_green_time.py taylor_green_time 5000 64 5 6 32 3
submit_problem ramansh_2d.py merge_vortices_easier 500 128 5 8 64 3
submit_problem ramansh_species_transport.py species_transport 10000 128 5 4 32 3
submit_problem ramansh_airfoil.py airfoil 5000 128 5 6 32 3

submit_problem ramansh_2d.py backward_facing_step_ood 100 128 5 8 16 3
submit_problem ramansh_taylor_green_time_coeffs2.py taylor_green_time_coeffs 5000 64 5 6 32 10
submit_problem ramansh_taylor_green_coeffs2.py taylor_green_coeffs 5000 64 4 6 32 10
