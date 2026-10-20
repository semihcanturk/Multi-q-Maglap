# SAT on High-level Synthesis (HLS). Hyperparameters in configs/general/hls/<target>/sat_*.yaml,
# from the paper's SAT tables (arXiv:2407.20912) except for a lower lr; the paper has no CP
# column, so the CP configs copy the DSP ones, as the GNN HLS configs do.
#
#   kernel=bi   -> sat_bigine (bidirected GIN structure extractor)
#   kernel=undi -> sat_gine   (undirected GIN structure extractor)
#
# HLS graphs have edge features, so one general config serves every PE. Two groups of runs:
# - the PEs of the GNN HLS sweeps (submit_hls_pe_sweep*.sh), for every target: No PE and
#   node+edge SPE (*_n_spe_e_spe) for Lap, multi-q Mag-PE with 5 potentials at q = 0.01 and
#   q = 0.1 (potentials k*q, k = 1..5), and PathLap;
# - the paper's own HLS setup, for DSP and LUT: node-only SPE (*_n_spe; App. B: "except for
#   High-level synthetic task where we find using z_node only has comparable performance")
#   for Lap and for Mag-PE with the paper's potentials, (1, 2)/100 for bidirected LUT and
#   (1..5)/100 otherwise.
#
# Runs every target x kernel by default; restrict with e.g.
#   TARGETS=lut KERNELS=bi SEEDS="121 122" bash run_hls_sat.sh
targets=${TARGETS:-dsp lut cp}
kernels=${KERNELS:-bi undi}
seeds=${SEEDS:-121 122 123 124 125}
device=0

for kernel in $kernels
do
for target in $targets
do

if [ "$kernel" = bi ]; then
    general_config=hls/$target/sat_bigine
    # paper's potentials for the bidirected SAT: (1, 2)/100 for LUT, (1..5)/100 otherwise
    if [ "$target" = lut ]; then paper_mq=maglap_2q001; else paper_mq=maglap_5q001; fi
else
    general_config=hls/$target/sat_gine
    paper_mq=maglap_5q001 # (1..5)/100 for every target
fi

# an empty pe config means No PE
pe_configs=(
    ""
    "lap10/lap_n_spe_e_spe"               # Lap
    "maglap10/maglap_5q001_n_spe_e_spe"   # Multi-q Mag-PE, q = 0.01
    "maglap10/maglap_5q_n_spe_e_spe"      # Multi-q Mag-PE, q = 0.1
    "pathlap10/pathlap_n_spe_e_spe"       # PathLap (not in the paper)
)
if [ "$target" != cp ]; then
    pe_configs+=(
        "lap10/lap_n_spe"                 # Lap, node SPE (paper)
        "maglap10/${paper_mq}_n_spe"      # Multi-q Mag-PE, node SPE (paper)
    )
fi

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
