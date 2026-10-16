"""Regression for experiment-only wrappers retaining old GPU KV entries."""
import weakref
import torch
from tools.worldstream_m3.experiment import ExperimentRuntime


def test_experiment_runtime_releases_without_cyclic_gc(llm,carriers,context):
    recorded=[]
    runtime=ExperimentRuntime(dict(mode='exact',offline_experiment=True),False,recorded)
    runtime.generate(llm,[torch.tensor([1,-200,4])],carriers,context=context,max_new_tokens=2)
    assert recorded and runtime.cache.entry is not None
    reference=weakref.ref(runtime)
    entry=weakref.ref(runtime.cache.entry)
    del runtime
    assert reference() is None and entry() is None
