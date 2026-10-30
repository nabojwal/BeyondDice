"""Windows-friendly Part 5 smoke CLI; all arguments configurable."""
import argparse
from dataclasses import fields
from train_engine import RunConfig, run_engine

def main():
    parser=argparse.ArgumentParser(description='MedSAM decoder-only training with W&B metrics')
    defaults=RunConfig()
    for f in fields(RunConfig):
        default=getattr(defaults,f.name)
        if isinstance(default,bool):
            parser.add_argument('--'+f.name.replace('_','-'),action='store_true',default=default)
        else:
            parser.add_argument('--'+f.name.replace('_','-'),type=type(default),default=default)
    args=parser.parse_args()
    run_engine(RunConfig(**vars(args)))

if __name__=='__main__':main()
