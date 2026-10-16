# SAT on the Open Circuit Benchmark (AMP), with the PE matrix of the paper's SAT rows
# (arXiv:2407.20912, Table 2; hyperparameters in configs/general/amp/<target>/sat_*.yaml).
#
#   kernel=bi   -> sat_bigin / sat_bigine (bidirected GIN structure extractor)
#   kernel=undi -> sat_gin / sat_gine     (undirected GIN structure extractor)
#
# AMP graphs have no edge features, so No PE and the node-only SignNet PEs run on the
# GIN/BIGIN configs, and PEs with an edge-level part (SPE, PathLap) on GINE/BIGINE.
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
    # paper's multi-q for the bidirected SAT: (1..10)/100 for BW, (1..5)/100 otherwise
    if [ "$target" = bw ]; then mq=maglap_10q001; else mq=maglap_5q001; fi
else
    gin=sat_gin
    gine=sat_gine
    mq=maglap_10q001 # (1..10)/100 for every target
fi

# "general config:pe config", an empty pe config meaning No PE
runs=(
    "$gin:"
    "$gin:lap10/lap_gnn"                    # Lap, SignNet
    "$gine:lap10/lap_n_spe_e_spe"           # Lap, SPE
    "$gin:maglap10/${mq}_gnn"               # Multi-q Mag-PE, SignNet
    "$gine:maglap10/${mq}_n_spe_e_spe"      # Multi-q Mag-PE, SPE
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
