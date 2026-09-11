#!/bin/bash
# Submits one sbatch job (via wrapper.sb) per (PE, objective) combination for the
# AMP benchmark, with the model config fixed at bigine_10q001. Each job runs the
# same 5 seeds as EDA_benchmark/run_amp.sh, one after another.
#
# Usage: ./submit_amp_pe_sweep.sh
set -e

cd "$(dirname "$0")"
mkdir -p logs

model_config=bigine_10q001
device=0
seeds=(121 122 123 124 125)
targets=(bw gain pm)

# "job-name-suffix:pe_config path (relative to EDA_benchmark/configs/pe/)"
pe_configs=(
  "lap_n_spe_e_spe:lap10/lap_n_spe_e_spe"
  "maglap_10q001_n_spe_e_spe:maglap10/maglap_10q001_n_spe_e_spe"
  "pathlap_n_spe_e_spe:pathlap10/pathlap_n_spe_e_spe"
)

for entry in "${pe_configs[@]}"; do
  pe_name=${entry%%:*}
  pe_config=${entry#*:}
  for target in "${targets[@]}"; do
    cmd="cd EDA_benchmark"
    for seed in "${seeds[@]}"; do
      cmd+=" && python main.py --general_config amp/${target}/${model_config} --pe_config ${pe_config} --seed ${seed} --device ${device}"
    done

    job_name="amp_${target}_${pe_name}"
    echo "Submitting ${job_name}"
    echo "  ${cmd}"
    sbatch --job-name="${job_name}" --output="logs/${job_name}_%j.out" wrapper.sb "${cmd}"
  done
done

echo "Submitted 9 jobs (3 PEs x 3 objectives, 5 seeds each)."
