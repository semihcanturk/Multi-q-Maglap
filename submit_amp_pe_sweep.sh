#!/bin/bash
# Submits one sbatch job (via wrapper.sb) per (PE, objective, train_stage_num)
# combination for the AMP benchmark, with the model config fixed at bigine_10q001.
# Each job runs the same 5 seeds as EDA_benchmark/run_amp.sh, one after another,
# then aggregates the 5 seeds' result_seed*.csv files into a mean+-std summary.csv
# (see aggregate_seed_results.py) written into the run's own train_files folder and
# printed at the end of the job's own SLURM log.
# train_stage_num selects which stage is in-distribution (2->3 vs 3->2
# generalization); AMP_data_processor.py caches both splits from one PE pass, and
# AMP_runner.py keys its output folder/wandb run name by train_stage_num so the two
# stage jobs for the same (PE, objective) can safely run concurrently.
#
# Usage: ./submit_amp_pe_sweep.sh
set -e

cd "$(dirname "$0")"
mkdir -p logs

model_config=bigat_10q001
device=0
seeds=(121 122 123 124 125)
targets=(bw gain pm)
train_stage_nums=(3)

# "job-name-suffix:pe_config path (relative to EDA_benchmark/configs/pe/)"
pe_configs=(
  "lap_n_spe_e_spe:lap10/lap_n_spe_e_spe"
  "maglap_10q001_n_spe_e_spe:maglap10/maglap_10q001_n_spe_e_spe"
  "maglap_10q_n_spe_e_spe:maglap10/maglap_10q_n_spe_e_spe"
  "pathlap_n_spe_e_spe:pathlap10/pathlap_n_spe_e_spe"
  # naive (linear) edge encoder: matched control for pathlap_n_spe_e_spe, plus the
  # edge-PE-only variant (no node-level SPE embedder).
  "pathlap_n_spe_e_naive:pathlap10/pathlap_n_spe_e_naive"
)

num_jobs=0
for entry in "${pe_configs[@]}"; do
  pe_name=${entry%%:*}
  pe_config=${entry#*:}
  for target in "${targets[@]}"; do
    for stage in "${train_stage_nums[@]}"; do
      cmd="cd EDA_benchmark"
      for seed in "${seeds[@]}"; do
        cmd+=" && python main.py --general_config amp/${target}/${model_config} --pe_config ${pe_config} --seed ${seed} --device ${device} --train_stage_num ${stage}"
      done
      cmd+=" && python aggregate_seed_results.py --general_config amp/${target}/${model_config} --pe_config ${pe_config} --train_stage_num ${stage}"

      job_name="amp_${target}_${pe_name}_stage${stage}"
      echo "Submitting ${job_name}"
      echo "  ${cmd}"
      sbatch --job-name="${job_name}" --output="logs/${job_name}_%j.out" wrapper.sb "${cmd}"
      num_jobs=$((num_jobs + 1))
    done
  done
done

echo "Submitted ${num_jobs} jobs (${#pe_configs[@]} PEs x ${#targets[@]} objectives x ${#train_stage_nums[@]} train_stage_nums, ${#seeds[@]} seeds each)."
