# SAT on High-level Synthesis (HLS), with the PE matrix of the paper's SAT rows
# (arXiv:2407.20912, Table 2; hyperparameters in configs/general/hls/<target>/sat_*.yaml).
#
#   kernel=bi   -> sat_bigine (bidirected GIN structure extractor)
#   kernel=undi -> sat_gine   (undirected GIN structure extractor)
#
# HLS graphs have edge features, so one general config serves every PE. SPE runs twice:
# node+edge SPE (*_n_spe_e_spe), as in the GINE/BIGINE HLS sweeps (submit_hls_pe_sweep*.sh),
# and node-only SPE (*_n_spe), the paper's HLS setup (App. B: "except for High-level
# synthetic task where we find using z_node only has comparable performance").
#
# Runs every target x kernel by default; restrict with e.g.
#   TARGETS=lut KERNELS=bi SEEDS="121 122" bash run_hls_sat.sh
targets=${TARGETS:-dsp lut}
kernels=${KERNELS:-bi undi}
seeds=${SEEDS:-121 122 123 124 125 126 127 128 129 130}
device=0

for kernel in $kernels
do
for target in $targets
do

if [ "$kernel" = bi ]; then
    general_config=hls/$target/sat_bigine
    # paper's multi-q for the bidirected SAT: (1, 2)/100 for LUT, (1..5)/100 for DSP
    if [ "$target" = lut ]; then mq=maglap_2q001; else mq=maglap_5q001; fi
else
    general_config=hls/$target/sat_gine
    mq=maglap_5q001 # (1..5)/100 for both targets
fi

# an empty pe config means No PE
pe_configs=(
    ""
    "lap10/lap_gnn"                    # Lap, SignNet
    "lap10/lap_n_spe_e_spe"            # Lap, node+edge SPE
    "maglap10/${mq}_gnn"               # Multi-q Mag-PE, SignNet
    "maglap10/${mq}_n_spe_e_spe"       # Multi-q Mag-PE, node+edge SPE
    "pathlap10/pathlap_n_spe_e_spe"    # PathLap, node+edge SPE (not in the paper)
    "lap10/lap_n_spe"                  # Lap, node SPE (paper)
    "maglap10/${mq}_n_spe"             # Multi-q Mag-PE, node SPE (paper)
)

for pe_config in "${pe_configs[@]}"
do
    pe_arg=${pe_config:+--pe_config $pe_config}
    for seed in $seeds
    do
        python main.py --general_config $general_config $pe_arg --seed $seed --device $device
    done
    python aggregate_seed_results.py --general_config $general_config $pe_arg
done

done
done
