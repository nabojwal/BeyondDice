#!/usr/bin/env python3
"""SSH/tmux orchestration for independent locked MedSAM runs; no SLURM/DDP.
Run from the MedSAM project root. Does not modify the training protocol.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# K5_seed1 is already active under the original launcher. Never schedule it here.
# Start larger K experiments earlier to shorten the critical path.
RUNS = [(k, s) for k in (20, 10, 5) for s in (1, 2, 3) if (k, s) != (5, 1)]
REQUIRED = ['run_locked.py', 'train_engine_locked.py', 'patient_manifests.py',
            'TrainData/images', 'TrainData/masks', 'label_schema.json',
            'manifests/patient_train_val.json', 'checkpoints/medsam_vit_b.pth']
STOP = False

def on_signal(signum, frame):
    global STOP
    STOP = True
    print(f'Received signal {signum}; no new jobs; active jobs remain alive until finished.', flush=True)

def gpu_memory():
    cp = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,memory.total',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    result = {}
    for line in cp.stdout.splitlines():
        a = [x.strip() for x in line.split(',')]
        if len(a) != 3:
            raise RuntimeError(f'Unexpected nvidia-smi output: {line}')
        result[int(a[0])] = (int(a[1]), int(a[2]))
    return result

def build_command(k, seed, root, resume=None, project='MedSAM-SPIDER-FewShot'):
    cmd = [sys.executable, '-u', 'run_locked.py', '--k', str(k), '--seed', str(seed),
        '--epochs', '100', '--patience', '15', '--min-delta', '0.001',
        '--val-frequency', '1', '--train-limit', '0', '--val-patients-limit', '0',
        '--val-examples-per-patient', '0', '--device', 'cuda:0',
        '--wandb-mode', 'online', '--wandb-project', project,
        '--experiment', f'K{k}_seed{seed}', '--output-root', str(root)]
    if resume:
        cmd.extend(['--resume', str(resume)])
    return cmd

def require_inputs():
    missing = [p for p in REQUIRED if not Path(p).exists()]
    if missing:
        raise RuntimeError('Missing project inputs: ' + ', '.join(missing))

def ensure_manifests():
    from patient_manifests import discover_pairs, prepare_split, fewshot_manifests
    records = discover_pairs('./TrainData/images', './TrainData/masks', r'(?P<patient>[^_]+)_t2')
    split = prepare_split(records, './manifests/patient_train_val.json', seed=42)
    assignments = split['assignments']
    ntrain = sum(v == 'train' for v in assignments.values())
    nval = sum(v == 'val' for v in assignments.values())
    if len(records) != 210 or ntrain != 168 or nval != 42:
        raise RuntimeError(f'Unexpected inventory/split: {len(records)} series, {ntrain} train, {nval} val')
    manifests = fewshot_manifests(split, './manifests/fewshot_train_val',
                                   seeds=[1, 2, 3], k_values=(5, 10, 20))
    for seed in (1, 2, 3):
        p5, p10, p20 = [set(manifests[(seed, k)]['patient_ids']) for k in (5, 10, 20)]
        if not (p5 <= p10 <= p20):
            raise RuntimeError(f'Non-nested manifests for seed {seed}')
        if p20 & {p for p, v in assignments.items() if v == 'val'}:
            raise RuntimeError('Train-validation overlap')
    return split, manifests

def recorded_completion(run_dir):
    marker = run_dir / 'cluster_completed.json'
    if not marker.is_file():
        return False
    state = json.loads(marker.read_text())
    if state.get('exit_code') != 0:
        return False
    return all((run_dir / p).is_file() for p in ('best.pth','last.pth','history.json','configuration.json'))

def save_status(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(temp, path)

def main():
    p = argparse.ArgumentParser(description='Schedule only the eight remaining runs; never K5_seed1')
    p.add_argument('--gpus', required=True, help='Comma-separated GPU IDs authorized for this project')
    p.add_argument('--execute', action='store_true', help='Actually launch the nine experiments')
    p.add_argument('--output-root', default='./runs')
    p.add_argument('--min-free-mib', type=int, default=18000,
                   help='Minimum free memory on GPU before starting a job')
    p.add_argument('--poll-seconds', type=int, default=30)
    p.add_argument('--wandb-project', default='MedSAM-SPIDER-FewShot')
    args = p.parse_args()
    gpus = [int(s) for s in args.gpus.split(',')]
    if not gpus or len(gpus) != len(set(gpus)) or args.min_free_mib < 1 or args.poll_seconds < 1:
        p.error('Invalid GPU list, memory threshold, or poll interval')
    require_inputs()
    current = gpu_memory()
    if any(i not in current for i in gpus):
        p.error(f'GPU IDs {gpus} not all present: {list(current)}')
    split, manifests = ensure_manifests()
    root = Path(args.output_root).resolve()
    if args.execute:
        root.mkdir(parents=True, exist_ok=True)
        lock_file = open(root / '.h100-remaining-launcher.lock', 'a+')
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Another launcher already holds the lock for this output root')
    print(f'Verified 210 series, split 168/42, nine nested manifests; scheduling 8 remaining runs. GPUs permitted: {gpus}', flush=True)
    jobs = []
    for k, seed in RUNS:
        name = f'K{k}_seed{seed}'
        directory = root / name
        if recorded_completion(directory):
            print('SKIP completed:', name, flush=True)
            continue
        if directory.exists():
            ckpt = directory / 'last.pth'
            if not ckpt.is_file():
                raise RuntimeError(f'{name}: existing directory without last.pth; inspect manually, no overwrite')
            if not (directory / 'configuration.json').is_file():
                raise RuntimeError(f'{name}: missing configuration.json; cannot safely resume')
            config = json.loads((directory / 'configuration.json').read_text())
            expected = {'k': k, 'seed': seed, 'epochs': 100, 'patience': 15,
                        'min_delta': 0.001, 'val_frequency': 1, 'wandb_mode': 'online',
                        'wandb_project': args.wandb_project, 'device': 'cuda:0',
                        'train_limit': 0, 'val_patients_limit': 0,
                        'val_examples_per_patient': 0, 'output_root': str(root), 'experiment': name}
            # The engine performs full provenance and complete config equality validation.
            # Reject clear mismatches here before submitting a process.
            for key, value in expected.items():
                if key not in config:
                    raise RuntimeError(f'{name}: missing configuration field {key}')
                if key == 'output_root' and Path(config[key]).resolve() == root:
                    continue
                if config[key] != value:
                    raise RuntimeError(f'{name}: saved configuration mismatch for {key}: {config[key]!r} != {value!r}')
            resume = ckpt
        else:
            resume = None
        jobs.append((k, seed, resume))
    for k, seed, resume in jobs:
        print(('RESUME' if resume else 'START'), f'K{k}_seed{seed}',
              ' '.join(build_command(k, seed, root, resume, args.wandb_project)), flush=True)
    if not args.execute:
        print('DRY RUN ONLY. Append --execute to launch.', flush=True)
        return 0
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    pending = list(jobs)
    active = {}
    failures = []
    while pending or active:
        for gpu, item in list(active.items()):
            process, k, seed, log_file, started = item
            rc = process.poll()
            if rc is None:
                continue
            log_file.close()
            directory = root / f'K{k}_seed{seed}'
            if rc == 0:
                if not all((directory / f).is_file() for f in ('last.pth','best.pth','history.json','configuration.json')):
                    rc = 99
                else:
                    save_status(directory / 'cluster_completed.json', {
                        'experiment': f'K{k}_seed{seed}', 'exit_code': 0,
                        'gpu': gpu, 'completed_unix': time.time()})
            if rc:
                failures.append(f'K{k}_seed{seed} (exit={rc})')
            print(f'EXIT K{k}_seed{seed}: {rc} on GPU {gpu}', flush=True)
            del active[gpu]
        if STOP:
            if pending:
                print(f'Stopping scheduling: {len(pending)} pending (active continue).', flush=True)
                pending.clear()
        if pending:
            free = gpu_memory()
            for gpu in gpus:
                if not pending or gpu in active or free[gpu][0] < args.min_free_mib:
                    continue
                k, seed, resume = pending.pop(0)
                name = f'K{k}_seed{seed}'
                directory = root / name
                log_dir = root / '.cluster_logs'
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = open(log_dir / f'{name}.log', 'a', buffering=1)
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                env['WANDB_MODE'] = 'online'
                cmd = build_command(k, seed, root, resume, args.wandb_project)
                print('LAUNCH', name, 'physical GPU', gpu, flush=True)
                log_file.write(f'\n=== Launch {time.strftime("%Y-%m-%d %H:%M:%S")} physical GPU {gpu}; resume={resume} ===\n')
                proc = subprocess.Popen(cmd, env=env, stdout=log_file,
                                        stderr=subprocess.STDOUT, cwd=os.getcwd())
                active[gpu] = (proc, k, seed, log_file, time.time())
        if active or pending:
            time.sleep(args.poll_seconds)
    print('Completed scheduling. Failures:', failures or 'none', flush=True)
    return 1 if failures else 0

if __name__ == '__main__':
    sys.exit(main())
