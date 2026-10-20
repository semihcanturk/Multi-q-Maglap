# SAT on the Open Circuit Benchmark (AMP): No PE and the SPE PEs (Lap, multi-q Mag-PE at two
# q values, PathLap). Hyperparameters in configs/general/amp/<target>/sat_*.yaml, from the
# paper's SAT tables (arXiv:2407.20912) except for a lower lr.
#
#   kernel=bi   -> sat_bigin / sat_bigine (bidirected GIN structure extractor)
#   kernel=undi -> sat_gin / sat_gine     (undirected GIN structure extractor)
#
# AMP graphs have no edge features, so No PE runs on the GIN/BIGIN configs, and the SPE PEs,
# which have an edge-level part, on GINE/BIGINE.
#
# Runs every target x kernel by default; restrict with e.g.
#   TARGETS=gain KERNELS=bi SEEDS="121 122" bash run_amp_sat.sh
targets=${TARGETS:-gain bw pm}
kernels=${KERNELS:-bi undi}
seeds=${SEEDS:-121 122 123 124 125}
device=0

for kernel in $kernels
do
for target in $targets
do

if [ "$kernel" = bi ]; then
    gin=sat_bigin
    gine=sat_bigine
    # number of potentials of the paper's bidirected SAT: 10 for BW, 5 otherwise
    if [ "$target" = bw ]; then mq=maglap_10q; else mq=maglap_5q; fi
else
    gin=sat_gin
    gine=sat_gine
    mq=maglap_10q # 10 potentials for every target
fi

# "general config:pe config", an empty pe config meaning No PE. The multi-q potentials are
# k*q for k = 1..#q: q = 0.01 is the paper's (1..5)/100 or (1..10)/100; maglap_5q has
# q = 0.1 and maglap_10q has q = 0.05, both up to 0.5.
runs=(
    "$gin:"
    "$gine:lap10/lap_n_spe_e_spe"           # Lap
    "$gine:maglap10/${mq}001_n_spe_e_spe"   # Multi-q Mag-PE, q = 0.01
    "$gine:maglap10/${mq}_n_spe_e_spe"      # Multi-q Mag-PE, q = 0.1 (5 potentials) / 0.05 (10)
    "$gine:pathlap10/pathlap_n_spe_e_spe"   # PathLap (not in the paper)
)

for run in "${runs[@]}"
do
    general_config=amp/$target/${run%%:*}
    pe_config=${run#*:}
    pe_arg=${pe_config:+--pe_config $pe_config}
    for seed in $seeds
    do
        python main.py --general_config $general_config $pe_arg --seed $seed --device $device
    done
    python aggregate_seed_results.py --general_config $general_config $pe_arg
done

done
done
