#!/bin/bash
# Submits one sbatch job (via wrapper.sb) per (PE, objective) combination for the
# HLS benchmark, with the model config fixed at bigine_maglap_n_5q_spe_001 (the only
# BIGINE general config available for HLS; --pe_config always overrides its baked-in
# PE settings, so the name doesn't constrain which PE actually runs). Each job runs
# the same 5 seeds as EDA_benchmark/run_hls.sh, one after another, then aggregates
# the 5 seeds' result_seed*.csv files into a mean+-std summary.csv (see
# aggregate_seed_results.py) written into the run's own train_files folder and
# printed at the end of the job's own SLURM log.
#
# HLS has no train_stage_num concept (that's AMP-specific), so unlike
# submit_amp_pe_sweep.sh there's no stage dimension here.
#
# Usage: ./submit_hls_pe_sweep.sh
set -e

cd "$(dirname "$0")"
mkdir -p logs

model_config=gat_maglap_n_5q_spe_001
device=0
seeds=(121 122 123 124 125)
targets=(dsp lut cp)

# "job-name-suffix:pe_config path (relative to EDA_benchmark/configs/pe/)"
pe_configs=(
  "lap_n_spe_e_spe:lap10/lap_n_spe_e_spe"
  "maglap_10q001_n_spe_e_spe:maglap10/maglap_5q001_n_spe_e_spe"
  "maglap_10q_n_spe_e_spe:maglap10/maglap_5q_n_spe_e_spe"
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
    cmd="cd EDA_benchmark"
    for seed in "${seeds[@]}"; do
      cmd+=" && python main.py --general_config hls/${target}/${model_config} --pe_config ${pe_config} --seed ${seed} --device ${device}"
    done
    cmd+=" && python aggregate_seed_results.py --general_config hls/${target}/${model_config} --pe_config ${pe_config}"

    job_name="hls_${target}_${pe_name}"
    echo "Submitting ${job_name}"
    echo "  ${cmd}"
    sbatch --account=aip-wolfg --job-name="${job_name}" --output="logs/${job_name}_%j.out" wrapper.sb "${cmd}"
    num_jobs=$((num_jobs + 1))
  done
done

echo "Submitted ${num_jobs} jobs (${#pe_configs[@]} PEs x ${#targets[@]} objectives, ${#seeds[@]} seeds each)."
