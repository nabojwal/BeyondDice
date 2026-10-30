"""Windows-compatible entry point for the locked MedSAM experiment protocol."""
import argparse
from dataclasses import fields
from train_engine_locked import RunConfig, run_engine

def main():
    parser=argparse.ArgumentParser(description='Locked nine-run MedSAM decoder-only protocol')
    defaults=RunConfig()
    for field in fields(RunConfig):
        value=getattr(defaults,field.name)
        flag='--'+field.name.replace('_','-')
        if isinstance(value,bool):
            parser.add_argument(flag,action='store_false' if value else 'store_true',default=value)
        else:
            parser.add_argument(flag,type=type(value),default=value)
    args=parser.parse_args()
    run_engine(RunConfig(**vars(args)))

if __name__=='__main__':main()
