"""Conservative measured-cost execution selection."""
class Scheduler:
    def __init__(self, config, costs):
        self.config, self.costs = config, costs

    def choose(self, key, updated, total, invalid_reason=None):
        if invalid_reason:
            return False, invalid_reason
        if updated == total:
            return False, 'insufficient_stable_windows'
        if self.config.mode == 'exact':
            return True, None
        return self.costs.beneficial(key, updated)

    def could_benefit(self, minimum_updates, block_indices):
        """M1 is only an upper bound on reuse; exact checks can add updates.

        This rejection-only gate never authorizes a sparse call or shares a cost
        between blocks. The eventual selection still needs its full exact key.
        """
        return any(r.get('block') in block_indices and r['updated']>=minimum_updates
                   and r.get('quality_certified',False)
                   and r['incremental_gpu_ms']<r['full_gpu_ms']
                   and r['incremental_wall_ms']<r['full_wall_ms'] for r in self.costs.records)
