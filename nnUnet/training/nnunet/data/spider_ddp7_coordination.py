"""Small coordination primitive, tested with real CPU/Gloo process groups."""
import torch.distributed as dist


def coordinated_call(action, *, root_only=False):
    """Propagate ordinary callback errors to every rank before the next phase.

    Callbacks must not contain distributed collectives. Process death, NCCL
    faults and errors inside upstream collectives remain the launcher's concern.
    """
    active = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if active else 0
    result, error = None, None
    if not root_only or rank == 0:
        try:
            result = action()
        except Exception as exc:
            error = f'rank {rank}: {type(exc).__name__}: {exc}'
    local = {'result': result, 'error': error}
    if active:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    failures = [item['error'] for item in gathered if item['error']]
    if failures:
        raise RuntimeError('Coordinated operation failed: ' + '; '.join(failures))
    return [item['result'] for item in gathered]
