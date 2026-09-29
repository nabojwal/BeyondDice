# Approved SPIDER baseline

`spider_baseline.json` records Dataset 11, fold 0, the 3d_fullres configuration,
seed 42 and the independently verified split hash/counts. The split installer and
trainer enforce the approved membership. Changing this file is an experimental
design change; do not change it underneath an existing run.

Architecture, loss, optimizer and scheduling are inherited from nnU-Net 2.8.1
and the existing nnUNetPlans.json. This file does not replace generated plans.
See [the canonical guide](../README.md) for commands.
