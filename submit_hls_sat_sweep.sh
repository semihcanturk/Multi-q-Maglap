#!/bin/bash
# Submits the SAT matrix of EDA_benchmark/run_hls_sat.sh with sbatch (via wrapper.sb), one job
# per (kernel, objective, PE): 2 kernels x (3 objectives x 5 PEs + 2 objectives x 2 paper PEs)
# = 38 jobs. Run it on the cluster's login node, from a checkout that has the SAT configs. Each
# job runs the 5 seeds one after another, then aggregate_seed_results.py folds the 5
# result_seed*.csv files into summary.csv, as submit_amp_sat_sweep.sh does.
# submit_hls_sat_sweep_cluv.sh submits the same jobs through cluv from your machine.
#
# The (kernel, objective) -> general config / PE mapping below is copied from run_hls_sat.sh;
# keep the three scripts in sync if any of them changes.
#
# Walltime: the GNN HLS runs took 1-2.7 h per seed (BiGAT, Rorqual), a 5-seed cluv HLS job hit
# a 16 h limit, and SAT's HLS run time hasn't been measured; hence the 24 h default. A job that
# hits its limit loses the seed in progress, the seeds after it and the aggregation (finished
# seeds keep their result_seed<N>.csv). Check the first jobs' time per seed and adjust WALLTIME.
#
# wrapper.sb runs each job in this checkout (cd $SLURM_SUBMIT_DIR, source .venv), not in a
# snapshot of one commit as cluv does: a job that starts after you edit or pull runs the new code.
#
# PE caches: every HLS objective and kernel reads the same cache directory per PE type, dims and
# q values (EDA_benchmark/data_processed/HLS/cdfg/processed_<pe>/), and a job that finds it
# missing builds it. Jobs that build the same cache at the same time write the same files, and a
# job can load a file that another is still writing. So the first job of each cache group is
# submitted as is, and the rest of the group with --dependency=after:<first>+CACHE_WAIT: they
# start no earlier than CACHE_WAIT minutes after the first one has started. A 5-potential
# Mag-PE cache for the ~19.6k HLS graphs takes ~4 min at the 13 ms/graph measured locally, hence
# the 15 min default. Set CACHE_WAIT=0 to submit without dependencies, e.g. when the caches
# already exist on this cluster. The job command also sets OPENBLAS_NUM_THREADS=1: with
# OpenBLAS's default threading, building a Mag-PE cache was ~20x slower in a local test.
#
# Usage: ./submit_hls_sat_sweep.sh
#        ACCOUNT=def-xyz GRES=gpu:h100:1 TARGETS="dsp lut" KERNELS=bi ./submit_hls_sat_sweep.sh
#        ACCOUNT= ./submit_hls_sat_sweep.sh    # submit without --account
set -e

cd "$(dirname "$0")"
# job logs of this submission get their own folder, so different sweeps don't mix
log_dir=logs/hls_sat_sweep_$(date +%Y-%m-%d_%H-%M-%S)

account=${ACCOUNT-aip-wolfg}   # set but empty = no --account
gres=${GRES:-}                 # empty = wrapper.sb's gpu:l40s:1
walltime=${WALLTIME:-12:00:00} # 5 seeds + aggregation in one job
cache_wait=${CACHE_WAIT:-5}   # minutes; 0 = no dependencies

device=0
seeds=(${SEEDS:-121 122 123 124 125})
targets=(${TARGETS:-dsp lut cp})
kernels=(${KERNELS:-bi undi})

sbatch_args=(--parsable --time="${walltime}")
if [ -n "${account}" ]; then sbatch_args+=(--account="${account}"); fi
if [ -n "${gres}" ]; then sbatch_args+=(--gres="${gres}"); fi

# "kernel:objective:general config:pe config:cache group", an empty pe config meaning No PE
tasks=()
for kernel in "${kernels[@]}"; do
  for target in "${targets[@]}"; do
    if [ "${kernel}" = bi ]; then
      general_config=hls/${target}/sat_bigine
      # paper's potentials for the bidirected SAT: (1, 2)/100 for LUT, (1..5)/100 otherwise
      if [ "${target}" = lut ]; then paper_mq=maglap_2q001; else paper_mq=maglap_5q001; fi
    elif [ "${kernel}" = undi ]; then
      general_config=hls/${target}/sat_gine
      paper_mq=maglap_5q001 # (1..5)/100 for every target
    else
      echo "Unknown kernel '${kernel}' (expected bi or undi)" >&2
      exit 1
    fi

    # The PEs of the GNN HLS sweeps (node+edge SPE; multi-q Mag-PE with 5 potentials at q = 0.01
    # and q = 0.1), then the paper's node-only SPE runs for DSP and LUT (the paper has no CP).
    pe_configs=(
      ""
      "lap10/lap_n_spe_e_spe"
      "maglap10/maglap_5q001_n_spe_e_spe"
      "maglap10/maglap_5q_n_spe_e_spe"
      "pathlap10/pathlap_n_spe_e_spe"
      "pathlap10/pathlap_n_spe_e_naive"
    )
    if [ "${target}" != cp ]; then
      pe_configs+=("lap10/lap_n_spe" "maglap10/${paper_mq}_n_spe")
    fi

    for pe_config in "${pe_configs[@]}"; do
      # configs that share a cache directory: same PE type, dims and q values
      case "${pe_config}" in
        "") cache=no_pe ;;
        maglap10/*) cache=${pe_config#maglap10/}; cache=${cache%%_n_*} ;; # maglap_5q001, maglap_5q, maglap_2q001
        *) cache=${pe_config%%/*} ;; # lap10, pathlap10
      esac
      tasks+=("${kernel}:${target}:${general_config}:${pe_config}:${cache}")
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
    cmd+=" && python main.py --general_config ${general_config}${pe_arg} --seed ${seed} --device ${device}"
  done
  cmd+=" && python aggregate_seed_results.py --general_config ${general_config}${pe_arg}"

  # job id of the first job submitted for this cache group, if any
  first_var=first_job_${cache}
  dep_args=()
  if [ "${cache_wait}" -gt 0 ] && [ -n "${!first_var}" ]; then
    dep_args=(--dependency="after:${!first_var}+${cache_wait}")
  fi

  job_name="hls_sat_${kernel}_${target}_${pe_name}"
  echo "Submitting ${job_name} ${dep_args[*]}"
  echo "  ${cmd}"
  job_id=$(sbatch "${sbatch_args[@]}" "${dep_args[@]}" --job-name="${job_name}" \
    --output="${log_dir}/${job_name}_%j.out" wrapper.sb "${cmd}")
  job_id=${job_id%%;*} # --parsable prints "id" or "id;cluster"
  echo "  -> job ${job_id}"
  if [ -z "${!first_var}" ]; then printf -v "${first_var}" '%s' "${job_id}"; fi
  num_jobs=$((num_jobs + 1))
done

echo "Submitted ${num_jobs} jobs (one per (kernel, objective, PE), ${#seeds[@]} seeds each). Logs: ${log_dir}/"
