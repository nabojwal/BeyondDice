"""Print or sequentially execute the 12 planned experiments. Pilot approval required."""
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='Actually launch experiments (default: print only)')
    parser.add_argument('--pilot-approved', action='store_true', help='Confirm complete K=1 pilot QC')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='offline')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    if args.execute and not args.pilot_approved:
        parser.error('Full matrix requires explicit --pilot-approved.')
    for seed in (42, 43, 44):
        for k in (1, 5, 10, 20):
            cmd = [sys.executable, str(Path(__file__).with_name('run_part6.py')),
                   '--k', str(k), '--seed', str(seed), '--epochs', str(args.epochs),
                   '--batch-size', str(args.batch_size), '--lr', '0.0001',
                   '--patience', '15', '--device', args.device,
                   '--wandb-mode', args.wandb_mode,
                   '--experiment', f'K{k}_seed{seed}']
            print(' '.join(f'"{x}"' if ' ' in x else x for x in cmd), flush=True)
            if args.execute:
                subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
