#!/bin/bash
# Per-seed variant of submit_hls_pe_sweep.sh: the same (PE, objective) matrix and model config,
# submitted with sbatch (via wrapper.sb) as one job per (PE, objective, seed) instead of one job
# running the 5 seeds back to back: PEs x objectives x 5 seeds jobs, each a single
# `python main.py`. Short jobs get through the queue sooner. Run it on the cluster's login node.
#
# The model config and PE list below are copied from submit_hls_pe_sweep.sh; keep the two in
# sync if either changes.
#
# Walltime: 4 h by default. The GNN HLS runs took 1-2.7 h per seed (BiGAT, Rorqual). A job that
# hits its limit loses that seed (training does not resume), so check the first jobs' run time
# before relying on the default.
#
# No aggregation step: the 5 seeds of a (PE, objective) finish separately. They all write their
# result_seed<N>.csv into the same train_files folder of this checkout (the seed is not part of
# that path; checkpoints are named model_seed<N>_*, so seeds don't clash), so once they have
# finished, run from EDA_benchmark/
#   python aggregate_seed_results.py --general_config hls/<target>/<model_config> --pe_config <pe>
#
# wrapper.sb runs each job in this checkout (cd $SLURM_SUBMIT_DIR, source .venv), not in a
# snapshot of one commit as cluv does: a job that starts after you edit or pull runs the new code.
#
# PE caches: every HLS objective reads the same cache directory per PE type, dims and q values
# (EDA_benchmark/data_processed/HLS/cdfg/processed_<pe>/), and a job that finds it missing builds
# it. Jobs that build the same cache at the same time write the same files, and a job can load a
# file that another is still writing. So the first job of each cache group is submitted as is,
# and the rest of the group with --dependency=after:<first>+CACHE_WAIT: they start no earlier
# than CACHE_WAIT minutes after the first one has started. The default is 30 min because the
# dir-RWPE cache stores a dense N x N x 12 array per graph (~13 GB for the CDFG splits), whose
# collation and saving take longer than the ~4 min of a Mag-PE cache; its build time on the
# cluster hasn't been measured. Set CACHE_WAIT=0 to submit without dependencies, e.g. when the
# caches already exist on this cluster. The job command also sets OPENBLAS_NUM_THREADS=1: with
# OpenBLAS's default threading, building a Mag-PE cache was ~20x slower in a local test.
#
# Usage: ./submit_hls_pe_sweep_per_seed.sh
#        ACCOUNT=def-xyz GRES=gpu:h100:1 TARGETS="dsp lut" ./submit_hls_pe_sweep_per_seed.sh
#        TARGETS=dsp SEEDS=121 ./submit_hls_pe_sweep_per_seed.sh   # 1 job per PE, to time a seed
set -e

cd "$(dirname "$0")"
# job logs of this submission get their own folder, so different sweeps don't mix
log_dir=logs/hls_pe_sweep_$(date +%Y-%m-%d_%H-%M-%S)

account=${ACCOUNT-aip-wolfg}   # set but empty = no --account
gres=${GRES:-}                 # empty = wrapper.sb's gpu:l40s:1
walltime=${WALLTIME:-4:00:00}  # one seed per job
cache_wait=${CACHE_WAIT:-30}   # minutes; 0 = no dependencies

model_config=${MODEL_CONFIG:-bigine_maglap_n_5q_spe_001}
device=0
seeds=(${SEEDS:-121 122 123 124 125})
targets=(${TARGETS:-dsp lut cp})

# "job-name-suffix:pe_config path (relative to EDA_benchmark/configs/pe/)"
pe_configs=(
  # "lap_n_spe_e_spe:lap10/lap_n_spe_e_spe"
  # "maglap_5q001_n_spe_e_spe:maglap10/maglap_5q001_n_spe_e_spe"
  # "maglap_5q_n_spe_e_spe:maglap10/maglap_5q_n_spe_e_spe"
  # "pathlap_n_spe_e_spe:pathlap10/pathlap_n_spe_e_spe"
  # "pathlap_n_spe_e_naive:pathlap10/pathlap_n_spe_e_naive"
  # directed random-walk PE: node + edge parts (matched to the n_spe_e_spe runs above), and
  # node part only.
  # "dirrwpe_n_e:dirrwpe/dirrwpe_n_e"
  "dirrwpe_n:dirrwpe/dirrwpe_n"
)

sbatch_args=(--parsable --time="${walltime}")
if [ -n "${account}" ]; then sbatch_args+=(--account="${account}"); fi
if [ -n "${gres}" ]; then sbatch_args+=(--gres="${gres}"); fi

# Check every config before submitting anything, so a stale checkout fails here, not in every job.
missing=0
for entry in "${pe_configs[@]}"; do
  for target in "${targets[@]}"; do
    for config in "general/hls/${target}/${model_config}" "pe/${entry#*:}"; do
      if [ ! -f "EDA_benchmark/configs/${config}.yaml" ]; then
        echo "Missing EDA_benchmark/configs/${config}.yaml" >&2
        missing=1
      fi
    done
  done
done
if [ "${missing}" -ne 0 ]; then
  echo "Update this checkout (git pull) and re-run; nothing was submitted." >&2
  exit 1
fi

mkdir -p "${log_dir}"
echo "Submitting from $(pwd) at $(git log -1 --format='%h %s'); job logs go to ${log_dir}/"
num_jobs=0
for entry in "${pe_configs[@]}"; do
  pe_name=${entry%%:*}
  pe_config=${entry#*:}
  # configs that share a cache directory: same PE type, dims and q values
  case "${pe_config}" in
    maglap10/*) cache=${pe_config#maglap10/}; cache=${cache%%_n_*} ;; # maglap_5q001, maglap_5q
    *) cache=${pe_config%%/*} ;; # lap10, pathlap10, dirrwpe
  esac

  for target in "${targets[@]}"; do
    for seed in "${seeds[@]}"; do
      cmd="export OPENBLAS_NUM_THREADS=1 && cd EDA_benchmark"
      cmd+=" && python main.py --general_config hls/${target}/${model_config} --pe_config ${pe_config} --seed ${seed} --device ${device}"

      # job id of the first job submitted for this cache group, if any
      first_var=first_job_${cache}
      dep_args=()
      if [ "${cache_wait}" -gt 0 ] && [ -n "${!first_var}" ]; then
        dep_args=(--dependency="after:${!first_var}+${cache_wait}")
      fi

      job_name="hls_${target}_${pe_name}_s${seed}"
      echo "Submitting ${job_name} ${dep_args[*]}"
      echo "  ${cmd}"
      job_id=$(sbatch "${sbatch_args[@]}" "${dep_args[@]}" --job-name="${job_name}" \
        --output="${log_dir}/${job_name}_%j.out" wrapper.sb "${cmd}")
      job_id=${job_id%%;*} # --parsable prints "id" or "id;cluster"
      echo "  -> job ${job_id}"
      if [ -z "${!first_var}" ]; then printf -v "${first_var}" '%s' "${job_id}"; fi
      num_jobs=$((num_jobs + 1))
    done
  done
done

echo "Submitted ${num_jobs} jobs (${#pe_configs[@]} PEs x ${#targets[@]} objectives x ${#seeds[@]} seeds, 1 seed per job). Logs: ${log_dir}/"
