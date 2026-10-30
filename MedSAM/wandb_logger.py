"""Optional W&B logging. Never uploads images, checkpoint files, or patient IDs."""
from __future__ import annotations
import warnings
from typing import Any


class WandbLogger:
    def __init__(self, *, mode: str, project: str, group: str, name: str,
                 config: dict[str, Any], tags: list[str], run_id: str | None = None):
        if mode not in {'online', 'offline', 'disabled'}:
            raise ValueError('wandb_mode must be online, offline, or disabled')
        self.run = None
        if mode == 'disabled':
            return
        try:
            import wandb
            kwargs = dict(project=project, group=group, name=name, tags=tags,
                          config=config, mode=mode, settings=wandb.Settings(init_timeout=30))
            if run_id:
                kwargs.update(id=run_id, resume='allow')
            self.run = wandb.init(**kwargs)
        except Exception as exc:
            warnings.warn(f'W&B disabled after initialization error: {exc}', RuntimeWarning)

    def log(self, stats: dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        try:
            flat = {}
            for key, value in stats.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    flat[key] = value
                elif key == 'per_structure_patient_mean_dice' and isinstance(value, dict):
                    for structure, dice in value.items():
                        flat[f'val_structure/{structure}_patient_mean_dice'] = dice
            self.run.log(flat, step=step)
        except Exception as exc:
            warnings.warn(f'W&B logging failed, local logging continues: {exc}', RuntimeWarning)

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as exc:
                warnings.warn(f'W&B finish failed: {exc}', RuntimeWarning)
