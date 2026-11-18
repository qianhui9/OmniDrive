"""Audit transformations against real source without loading weights or CUDA."""
import ast
import copy
from pathlib import Path
import pytest
from tools.component_latency_eval.instrumentation import StripScopes, transform

CASES = [
    ('worldstream/m1/state_manager.py','WorldStateManager','precheck_with_observation',('OGSM.Precheck',False,'OGSM.Total')),
    ('worldstream/m1/state_manager.py','WorldStateManager','commit',('OGSM.Commit',False,'OGSM.Total')),
    ('worldstream/m2/runtime.py','IncrementalVisionRuntime','_forward_frame','planning'),
    ('worldstream/m2/runtime.py','IncrementalVisionRuntime','_execute','prefix'),
    ('worldstream/m2/runtime.py','IncrementalVisionRuntime','forward_from_m1',('DAVP.Total',True,'')),
    ('worldstream/m2/runtime.py','IncrementalVisionRuntime','_update_global',('DAVP.Global Execution',True,'DAVP.Total')),
    ('projects/mmdet3d_plugin/models/detectors/petr3d.py','Petr3D','simple_test_pts','perception'),
    ('worldstream/m3/runtime.py','M3Runtime','_decode_current_frame','queries'),
    ('worldstream/m3/prefill_executor.py','PrefillExecutor','generate',('QCSE.Target Generation',True,'QCSE.Total')),
]


@pytest.mark.parametrize('path,cls,fn,kind',CASES)
def test_original_statements_and_valid_compilation(path, cls, fn, kind):
    module = ast.parse(Path(path).read_text())
    # Resolve method by name, requiring exactly one match in source.
    methods = [n for c in module.body if isinstance(c,ast.ClassDef) for n in c.body
               if isinstance(n,ast.FunctionDef) and n.name==fn]
    assert len(methods)==1
    original = ast.Module(body=[copy.deepcopy(methods[0])],type_ignores=[])
    original.body[0].decorator_list=[]
    tree = transform(copy.deepcopy(original),kind)
    recovered=StripScopes().visit(copy.deepcopy(tree))
    assert ast.dump(original,include_attributes=False)==ast.dump(recovered,include_attributes=False)
    compile(ast.fix_missing_locations(tree),path,'exec')
    # No newly inserted scopes inside the actual token loop or EVA layer loop.
    for node in ast.walk(tree):
        if isinstance(node,ast.While) or (isinstance(node,ast.For) and 'self.model.blocks' in ast.unparse(node.iter)):
            assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='__component_span' for n in ast.walk(node))
