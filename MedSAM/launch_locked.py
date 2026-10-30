"""Fail-closed nine-run launcher. Dry-run by default, pilot-first gate."""
import argparse
import subprocess
import sys
from pathlib import Path

SHOTS=(5,10,20)
SEEDS=(1,2,3)

def schedule():
    return [(k,seed) for k in SHOTS for seed in SEEDS]

def command(k,seed,mode,device,root):
    return [sys.executable,'run_locked.py','--k',str(k),'--seed',str(seed),
        '--epochs','100','--patience','15','--min-delta','0.001','--val-frequency','1',
        '--train-limit','0','--val-patients-limit','0','--val-examples-per-patient','0',
        '--wandb-mode',mode,'--device',device,'--experiment',f'K{k}_seed{seed}',
        '--output-root',str(root)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--execute',action='store_true')
    p.add_argument('--pilot-approved',action='store_true')
    p.add_argument('--pilot-only',action='store_true')
    p.add_argument('--skip-verified-pilot',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--wandb-mode',choices=('online','offline','disabled'),default='offline')
    p.add_argument('--output-root',default='./runs')
    a=p.parse_args()
    if a.pilot_only and a.skip_verified_pilot:p.error('Choose pilot-only or skip-verified-pilot, not both')
    runs=schedule()[:1] if a.pilot_only else schedule()
    if a.skip_verified_pilot:
        pilot=Path(a.output_root)/'K5_seed1'
        if not all((pilot/f).is_file() for f in ('best.pth','last.pth','history.json','configuration.json')):
            p.error('Cannot skip pilot: missing required local pilot artifacts')
        runs=[r for r in runs if r!=(5,1)]
    if a.execute and not a.pilot_only and not a.pilot_approved:
        p.error('Full matrix execution requires --pilot-approved after pilot verification')
    for k,seed in runs:
        path=Path(a.output_root)/f'K{k}_seed{seed}'
        cmd=command(k,seed,a.wandb_mode,a.device,Path(a.output_root))
        
        assert all(isinstance(arg, str) for arg in cmd)

        i = cmd.index("--val-examples-per-patient")

        assert cmd[i + 1] == "0"
        assert cmd[i + 2] == "--wandb-mode"
        assert cmd[i + 3] == a.wandb_mode

        print(
            f"ARGUMENT CHECK: K={k}, seed={seed}, PASS",
            flush=True,
        )

        # print('COMMAND:',subprocess.list2cmdline(cmd),flush=True)
        print("COMMAND:", " ".join(
            subprocess.list2cmdline([str(arg)]) for arg in cmd
        ), flush=True)
        if a.execute:
            if path.exists():
                raise FileExistsError(f'Existing experiment {path}; resume separately after auditing provenance')
            subprocess.run(cmd,check=True)

if __name__=='__main__':main()
