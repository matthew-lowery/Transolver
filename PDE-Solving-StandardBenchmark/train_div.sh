#!/bin/bash

set -euo pipefail

DIV_LOSS_WEIGHTS=(1.0)
SEEDS=(1 2 3)

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

for div_loss_weight in "${DIV_LOSS_WEIGHTS[@]}"; do
for seed in 1; do
    sp "python3 ramansh_2d.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=flow_cylinder_laminar --ntrain=100 --n-hidden=128 --n-layers=5 --n-heads=8 --slice-num=32" 3 "div_flow_cylinder_laminar"

    sp "python3 ramansh_2d.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=flow_cylinder_shedding --ntrain=10000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32" 3 "div_flow_cylinder_shedding"

    sp "python3 ramansh_2d.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=lid_cavity_flow --ntrain=10000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=16" 3 "div_lid_cavity_flow"

    sp "python3 ramansh_2d.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=backward_facing_step --ntrain=500 --n-hidden=128 --n-layers=5 --n-heads=8 --slice-num=16" 3 "div_backward_facing_step"

    sp "python3 ramansh_2d.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=buoyancy_cavity_flow --ntrain=10000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32" 3 "div_buoyancy_cavity_flow"

    sp "python3 ramansh_2d.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_exact --ntrain=5000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=64" 3 "div_taylor_green_exact"

    sp "python3 ramansh_taylor_green_time.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_time --ntrain=5000 --n-hidden=64 --n-layers=5 --n-heads=6 --slice-num=32" 3 "div_taylor_green_time"

    sp "python3 ramansh_2d.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=merge_vortices_easier --ntrain=500 --n-hidden=128 --n-layers=5 --n-heads=8 --slice-num=64" 3 "div_merge_vortices_easier"

    sp "python3 ramansh_species_transport.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=species_transport --ntrain=10000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32" 3 "div_species_transport"

    sp "python3 ramansh_airfoil.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=airfoil --ntrain=5000 --n-hidden=128 --n-layers=5 --n-heads=6 --slice-num=32" 3 "div_airfoil"

    sp "python3 ramansh_2d.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=backward_facing_step_ood --ntrain=100 --n-hidden=128 --n-layers=5 --n-heads=8 --slice-num=16" 3 "div_backward_facing_step_ood"

    sp "python3 ramansh_taylor_green_time_coeffs2.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_time_coeffs --ntrain=5000 --n-hidden=64 --n-layers=5 --n-heads=6 --slice-num=32" 10 "div_taylor_green_time_coeffs"

    sp "python3 ramansh_taylor_green_coeffs2.py --project-name=transolver_div_loss --div-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/transolver-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_coeffs --ntrain=5000 --n-hidden=64 --n-layers=4 --n-heads=6 --slice-num=32" 10 "div_taylor_green_coeffs"
    done
done
