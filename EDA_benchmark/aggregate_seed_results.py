"""Aggregate a sweep job's per-seed result_seed*.csv files into mean +- std.

Each AMP/HLS run writes its own runner/runner/train_files/.../result_seed{seed}.csv
(see AMP_runner.py / HLS_runner.py). This script locates that folder for a given
(general_config, pe_config[, train_stage_num]) combination -- the same args used to
launch the runs -- reads every result_seed*.csv in it, and writes a summary.csv with
the mean and std of every metric column, grouped by test-set row, next to them.

Usage (same args as main.py, run after all seeds finish):
    python aggregate_seed_results.py --general_config amp/gain/bigine_10q001 \\
        --pe_config pathlap10/pathlap_n_spe_e_spe --train_stage_num 2
"""
import argparse
import glob
import importlib
import os

import pandas as pd

from utils import load_config, merge_dicts


def main():
    parser = argparse.ArgumentParser(description='aggregate per-seed results into mean+-std')
    parser.add_argument('--general_config', required=True)
    parser.add_argument('--pe_config', default=None)
    parser.add_argument('--train_stage_num', type=int, default=None, choices=[2, 3])
    args = parser.parse_args()

    general_config = load_config('./configs/general/'+str(args.general_config)+'.yaml')
    if args.pe_config is not None:
        pe_config = load_config('./configs/pe/'+str(args.pe_config)+'.yaml')
        config = merge_dicts(general_config, pe_config)
    else:
        config = general_config
    if args.train_stage_num is not None:
        config['task']['train_stage_num'] = args.train_stage_num
    # only used by the runner to name its own per-seed file; irrelevant here since
    # we glob for every seed's file regardless of this value.
    config['utils']['seed'] = 0

    runner_path = 'runner.'+str(config['task']['name'])+'_runner'
    runner_name = config['task']['name']+'Runner'
    module = importlib.import_module(runner_path)
    cls = getattr(module, runner_name)
    runner = cls(config)  # cheap: only builds folder paths, no dataset/model loading

    result_files = sorted(glob.glob(os.path.join(runner.train_folder, 'result_seed*.csv')))
    if not result_files:
        print(f'No result_seed*.csv files found in {runner.train_folder}')
        return

    frames = []
    for f in result_files:
        df = pd.read_csv(f)
        df['seed'] = os.path.basename(f)[len('result_seed'):-len('.csv')]
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    metric_cols = [c for c in all_df.columns if c not in ('test set name', 'seed')]
    summary = all_df.groupby('test set name')[metric_cols].agg(['mean', 'std'])

    summary_path = os.path.join(runner.train_folder, 'summary.csv')
    summary.to_csv(summary_path)

    print(f'\nAggregated {len(result_files)} seed(s) from {runner.train_folder}:')
    print(f'  seeds: {sorted(all_df["seed"].unique().tolist())}')
    print(summary.to_string())
    print(f'\nWrote {summary_path}')


if __name__ == '__main__':
    main()
