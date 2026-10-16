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
