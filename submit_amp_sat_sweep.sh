#!/bin/bash
# Submits the SAT matrix of EDA_benchmark/run_amp_sat.sh with sbatch (via wrapper.sb), one job
# per (kernel, objective, PE): 2 kernels x 3 objectives x 5 PEs = 30 jobs. Run it on the
# cluster's login node, from a checkout that has the SAT configs. Each job runs the 5 seeds one
# after another, then aggregate_seed_results.py folds the 5 result_seed*.csv files into
# summary.csv. submit_amp_sat_sweep_cluv.sh submits the same jobs through cluv from your machine.
#
# The (kernel, objective) -> general config / multi-q mapping below is copied from
# run_amp_sat.sh; keep the three scripts in sync if any of them changes.
#
# wrapper.sb runs each job in this checkout (cd $SLURM_SUBMIT_DIR, source .venv), not in a
# snapshot of one commit as cluv does: a job that starts after you edit or pull runs the new code.
#
# PE caches: every AMP objective and kernel reads the same cache directory per PE
# (EDA_benchmark/data_processed/AMP/processed_<pe>/), and a job that finds it missing builds it.
# Jobs that build the same cache at the same time write the same files, and a job can load a
# file that another is still writing. So the first job of each cache group is submitted as is,
# and the rest of the group with --dependency=after:<first>+CACHE_WAIT: they start no earlier
# than CACHE_WAIT minutes after the first one has started. Set CACHE_WAIT=0 to submit without
# dependencies, e.g. when the caches already exist on this cluster. The job command also sets
# OPENBLAS_NUM_THREADS=1 (as scripts/job.sh does for cluv jobs): with OpenBLAS's default
# threading, building a Mag-PE cache was ~20x slower in a local test.
#
# Usage: ./submit_amp_sat_sweep.sh
#        ACCOUNT=def-xyz GRES=gpu:h100:1 TARGETS="gain bw" KERNELS=bi ./submit_amp_sat_sweep.sh
#        ACCOUNT= ./submit_amp_sat_sweep.sh    # submit without --account
set -e

cd "$(dirname "$0")"
# job logs of this submission get their own folder, so different sweeps don't mix
log_dir=logs/amp_sat_sweep_$(date +%Y-%m-%d_%H-%M-%S)

account=${ACCOUNT-aip-wolfg}   # set but empty = no --account
gres=${GRES:-}                 # empty = wrapper.sb's gpu:l40s:1
walltime=${WALLTIME:-3:00:00} # 5 seeds + aggregation in one job
cache_wait=${CACHE_WAIT:-5}   # minutes; 0 = no dependencies

device=0
seeds=(${SEEDS:-121 122 123 124 125})
targets=(${TARGETS:-gain bw pm})
kernels=(${KERNELS:-bi undi})
train_stage_num=${TRAIN_STAGE_NUM:-3}

sbatch_args=(--parsable --time="${walltime}")
if [ -n "${account}" ]; then sbatch_args+=(--account="${account}"); fi
if [ -n "${gres}" ]; then sbatch_args+=(--gres="${gres}"); fi

# "kernel:objective:general config:pe config:cache group", an empty pe config meaning No PE
tasks=()
for kernel in "${kernels[@]}"; do
  for target in "${targets[@]}"; do
    if [ "${kernel}" = bi ]; then
      gin=sat_bigin
      gine=sat_bigine
      # number of potentials of the paper's bidirected SAT: 10 for BW, 5 otherwise
      if [ "${target}" = bw ]; then mq=maglap_10q; else mq=maglap_5q; fi
    elif [ "${kernel}" = undi ]; then
      gin=sat_gin
      gine=sat_gine
      mq=maglap_10q # 10 potentials for every target
    else
      echo "Unknown kernel '${kernel}' (expected bi or undi)" >&2
      exit 1
    fi

    # AMP graphs have no edge features, so No PE runs on GIN/BIGIN and the SPE PEs, which
    # have an edge part, on GINE/BIGINE. Multi-q Mag-PE at q = 0.01 and at q = 0.1 (5q) /
    # 0.05 (10q); the potentials are k*q for k = 1..#q.
    runs=(
      "${gin}:"
      "${gine}:lap10/lap_n_spe_e_spe"
      "${gine}:maglap10/${mq}001_n_spe_e_spe"
      "${gine}:maglap10/${mq}_n_spe_e_spe"
      "${gine}:pathlap10/pathlap_n_spe_e_spe"
      "${gine}:pathlap10/pathlap_n_spe_e_naive"
    )

    for run in "${runs[@]}"; do
      pe_config=${run#*:}
      # each PE config above has its own cache directory (PE type, dims and q values), shared
      # by every kernel and objective that uses it
      cache=${pe_config##*/}
      cache=${cache:-no_pe}
      tasks+=("${kernel}:${target}:amp/${target}/${run%%:*}:${pe_config}:${cache}")
    done
  done
done

# Check every config before submitting anything, so a stale checkout fails here, not in every job.
missing=0
for task in "${tasks[@]}"; do
  IFS=: read -r kernel target general_config pe_config cache <<< "${task}"
  for config in "general/${general_config}" ${pe_config:+"pe/${pe_config}"}; do
    if [ ! -f "EDA_benchmark/configs/${config}.yaml" ]; then
      echo "Missing EDA_benchmark/configs/${config}.yaml" >&2
      missing=1
    fi
  done
done
if [ "${missing}" -ne 0 ]; then
  echo "Update this checkout (git pull) and re-run; nothing was submitted." >&2
  exit 1
fi

mkdir -p "${log_dir}"
echo "Submitting from $(pwd) at $(git log -1 --format='%h %s'); job logs go to ${log_dir}/"
num_jobs=0
for task in "${tasks[@]}"; do
  IFS=: read -r kernel target general_config pe_config cache <<< "${task}"
  pe_arg=${pe_config:+ --pe_config ${pe_config}}
  pe_name=${pe_config##*/}
  pe_name=${pe_name:-no_pe}

  cmd="export OPENBLAS_NUM_THREADS=1 && cd EDA_benchmark"
  for seed in "${seeds[@]}"; do
    cmd+=" && python main.py --general_config ${general_config}${pe_arg} --seed ${seed} --device ${device} --train_stage_num ${train_stage_num}"
  done
  cmd+=" && python aggregate_seed_results.py --general_config ${general_config}${pe_arg} --train_stage_num ${train_stage_num}"

  # job id of the first job submitted for this cache group, if any
  first_var=first_job_${cache}
  dep_args=()
  if [ "${cache_wait}" -gt 0 ] && [ -n "${!first_var}" ]; then
    dep_args=(--dependency="after:${!first_var}+${cache_wait}")
  fi

  job_name="amp_sat_${kernel}_${target}_${pe_name}_stage${train_stage_num}"
  echo "Submitting ${job_name} ${dep_args[*]}"
  echo "  ${cmd}"
  job_id=$(sbatch "${sbatch_args[@]}" "${dep_args[@]}" --job-name="${job_name}" \
    --output="${log_dir}/${job_name}_%j.out" wrapper.sb "${cmd}")
  job_id=${job_id%%;*} # --parsable prints "id" or "id;cluster"
  echo "  -> job ${job_id}"
  if [ -z "${!first_var}" ]; then printf -v "${first_var}" '%s' "${job_id}"; fi
  num_jobs=$((num_jobs + 1))
done

echo "Submitted ${num_jobs} jobs (${#kernels[@]} kernels x ${#targets[@]} objectives x ${#runs[@]} PEs, ${#seeds[@]} seeds each). Logs: ${log_dir}/"
