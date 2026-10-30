"""PowerShell-friendly Part 6 CLI. One invocation performs exactly one experiment."""
import argparse
from dataclasses import fields
from train_engine_part6 import RunConfig, run_engine


def main() -> None:
    defaults = RunConfig()
    parser = argparse.ArgumentParser(description='MedSAM full decoder-only few-shot training')
    for field in fields(RunConfig):
        default = getattr(defaults, field.name)
        flag = '--' + field.name.replace('_', '-')
        if isinstance(default, bool):
            parser.add_argument(flag, action='store_false' if default else 'store_true', default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    args = parser.parse_args()
    run_engine(RunConfig(**vars(args)))


if __name__ == '__main__':
    main()
