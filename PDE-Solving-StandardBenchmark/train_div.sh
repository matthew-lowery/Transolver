#!/bin/bash

set -euo pipefail

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
#SBATCH --account=bgcs-delta-gpu
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

for seed in 2 3; do
    div_loss_weight=0.01
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=flow_cylinder_laminar --ntrain=100 --npoints=1000" 2 "div_flow_cylinder_laminar"

    div_loss_weight=0.001
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=flow_cylinder_shedding --ntrain=10000 --npoints=1000" 2 "div_flow_cylinder_shedding"

    div_loss_weight=0.001
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=lid_cavity_flow --ntrain=10000 --npoints=1000" 2 "div_lid_cavity_flow"

    div_loss_weight=0.01
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=buoyancy_cavity_flow --ntrain=10000 --npoints=5000" 6 "div_buoyancy_cavity_flow"

    div_loss_weight=0.1
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green --ntrain=5000 --npoints=500" 2 "div_taylor_green"

    div_loss_weight=0.1
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_coeffs --ntrain=5000 --npoints=500" 10 "div_taylor_green_coeffs"

    div_loss_weight=0.01
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_spacetime --ntrain=5000 --npoints=500" 2 "div_taylor_green_spacetime"

    div_loss_weight=0.1
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_spacetime_coeffs --ntrain=5000 --npoints=500" 10 "div_taylor_green_spacetime_coeffs"

    div_loss_weight=0.1
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=backward_facing_step --ntrain=500 --npoints=1000" 2 "div_backward_facing_step"

    div_loss_weight=0.001
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=merge_vortices_easier --ntrain=500 --npoints=500" 2 "div_merge_vortices_easier"

    div_loss_weight=0.001
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=species_transport --ntrain=10000 --npoints=7000" 10 "div_species_transport"

    div_loss_weight=0.001 # provisional: inherited from species transport
    sp "python3 ramansh_train.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=forced_turb --ntrain=10000 --npoints=7000" 10 "div_forced_turb"
done
