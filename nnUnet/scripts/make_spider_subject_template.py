"""Create a subject-map template. Fill/verify IDs against subject-level metadata."""
import argparse
import csv
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--suggest-filename-prefix',action='store_true',help='Draft suggestions only; verify these are actual subject IDs')
a=p.parse_args()
cases=sorted(json.loads((a.raw/'spider_source_manifest.json').read_text())['test'])
if len(cases)!=87:raise ValueError('Expected 87 test scans')
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['case_id','subject_id'])
    w.writerows((c,c.split('_')[0] if a.suggest_filename_prefix else '') for c in cases)
print('Template written. Verify subject IDs before evaluation; filename prefixes alone do not prove patient identity.')
