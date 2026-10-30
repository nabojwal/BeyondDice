"""Retired metadata writer; the prepared Dataset011 metadata is already validated."""


def main():
    """Prevent the old import-time write to an obsolete dataset path."""
    raise SystemExit(
        'Retired: Dataset011 metadata already exists. Follow training/nnunet/README.md. '
        'The original writer is preserved in outputs/integration_archive/json_create.py.')


if __name__ == '__main__':
    main()
