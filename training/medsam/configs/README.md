# MedSAM experiment configurations

Store versioned JSON experiment settings here when fine-tuning is implemented.
Record data and checkpoint paths, split definition, seed, selected classes,
prompt policy, trainable components, batch size, optimizer, learning rate,
epochs, and validation settings.

Use repository-relative paths where possible. A future entry point must
explicitly load these files. Store checkpoint binaries in the ignored weight
or run directories.
