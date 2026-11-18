"""Experiment-local AST scopes; original functions/statements run in original order.

Only __code__ is temporarily replaced, preserving identities used by runtime
cache signatures. Inference computation order and streams are unchanged. The
existing post-QCSE export can be queued until after the forward timing fence.
"""
import ast
import copy
import hashlib
import inspect
import textwrap


def scope(body, name, **keywords):
    call = ast.Call(func=ast.Name(id='__component_span', ctx=ast.Load()),
                    args=[ast.Constant(name)],
                    keywords=[ast.keyword(arg=k, value=ast.Constant(v)) for k, v in keywords.items()])
    return ast.With(items=[ast.withitem(context_expr=call)], body=body)


def starts(node, prefix):
    return ast.unparse(node).startswith(prefix)


def interval(body, first, last, name, **kwargs):
    a = next(i for i, n in enumerate(body) if starts(n, first))
    b = next(i for i, n in enumerate(body) if i >= a and starts(n, last))
    return body[:a] + [scope(body[a:b], name, **kwargs)] + body[b:]


class StripScopes(ast.NodeTransformer):
    def visit_If(self, node):
        node = self.generic_visit(node)
        for field in ('body', 'orelse'):
            statements = getattr(node, field)
            restored = []
            i = 0
            while i < len(statements):
                statement = statements[i]
                if isinstance(statement, ast.FunctionDef) and statement.name == '__component_output_export':
                    call = statements[i+1]
                    if not (isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
                            and isinstance(call.value.func, ast.Name)
                            and call.value.func.id == '__component_defer_export'):
                        raise RuntimeError('Unrecognized export deferral')
                    restored.extend(statement.body)
                    i += 2
                else:
                    restored.append(statement)
                    i += 1
            setattr(node, field, restored)
        return node

    def visit_With(self, node):
        node = self.generic_visit(node)
        call = node.items[0].context_expr
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == '__component_span':
            return node.body
        return node


def defer_output_export(fn):
    """Capture the unchanged export tail, after all actual M3 generation.

    The returned generated_text list is filled in-place by the callback. Full
    WorldStream execution-contract checks consume M3 diagnostics, not this list.
    """
    found = 0
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        candidates = [i for i, stmt in enumerate(node.body) if starts(stmt, 'export_ms = 0.')]
        if not candidates:
            continue
        start = candidates[0]
        if not starts(node.body[-1], 'self.language_export_wall_ms ='):
            raise RuntimeError('Output export boundary changed')
        callback = ast.FunctionDef(name='__component_output_export',
            args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=node.body[start:], decorator_list=[])
        enqueue = ast.Expr(value=ast.Call(func=ast.Name(id='__component_defer_export', ctx=ast.Load()),
            args=[ast.Name(id='self', ctx=ast.Load()), ast.Name(id='m3', ctx=ast.Load()),
                  ast.Name(id='__component_output_export', ctx=ast.Load())], keywords=[]))
        node.body = node.body[:start] + [callback, enqueue]
        found += 1
    if found != 1:
        raise RuntimeError('Expected exactly one real post-QCSE output export tail')


def transform(tree, kind):
    fn = tree.body[0]
    fn.decorator_list = []
    if kind == 'planning':
        fn.body = interval(fn.body, 'input_finite =', 'execution_start =',
                           'DAVP.Visual Planning', gpu=True, parent='DAVP.Total')
        fn.body = interval(fn.body, 'self.last_diagnostics =', 'x0 =',
                           'DAVP.Visual Planning', gpu=True, parent='DAVP.Total')
    elif kind == 'prefix':
        fn.body = interval(fn.body, 'stages =', 'global_context =',
                           'DAVP.Incremental EVA', gpu=True, parent='DAVP.Total')
    elif kind in ('perception', 'export_only'):
        # Only actual head branches; shared ROI/position preparation remains E2E residual.
        defer_output_export(fn)
        found = 0
        for node in fn.body:
            if isinstance(node, ast.If) and ast.unparse(node.test) in ('self.with_pts_bbox', 'self.with_map_head'):
                name = 'Detection Head' if ast.unparse(node.test) == 'self.with_pts_bbox' else 'Mapping Head'
                if kind == 'perception':
                    node.body = [scope(node.body, name, gpu=True, parent='Original Perception')]
                found += 1
        if found != 2:
            raise RuntimeError('Perception boundary changed')
    elif kind == 'queries':
        loop = next(n for n in fn.body if isinstance(n, ast.For) and 'ids_list' in ast.unparse(n.iter))
        class Direct(ast.NodeTransformer):
            def visit_If(self, node):
                node = self.generic_visit(node)
                if "proposal['admitted']" in ast.unparse(node.test):
                    node.body = [scope(node.body, 'QCSE.Direct Answer', gpu=True, path='direct_answer', parent='QCSE.Total')]
                return node
        loop = Direct().visit(loop)
        loop.body = [scope(loop.body, 'Query E2E', gpu=True, sync=True, query=True)]
    else:
        name, gpu, parent = kind
        fn.body = [scope(fn.body, name, gpu=gpu, parent=parent)]
    return tree


class Instrumentation:
    def __init__(self, model, recorder, exports=None, components=True):
        self.recorder = recorder
        self.exports = exports
        self.saved, self.globals, self.audit = [], {}, []
        from worldstream.m3.answer_validity import propose_fact_answer
        specs = [
            (model.m1_manager.precheck_with_observation, ('OGSM.Precheck', False, 'OGSM.Total')),
            (model.m1_manager.commit, ('OGSM.Commit', False, 'OGSM.Total')),
            (model.m2_runtime.forward_from_m1, ('DAVP.Total', True, '')),
            (model.m2_runtime.model.patch_embed.forward, ('DAVP.Patch Embedding', True, 'DAVP.Total')),
            (model.m2_runtime._forward_frame, 'planning'),
            (model.m2_runtime._execute, 'prefix'),
            (model.m2_runtime._update_global, ('DAVP.Global Execution', True, 'DAVP.Total')),
            (model.simple_test_pts, 'perception'),
            (model.m3_runtime.generate, ('QCSE.Total', True, '')),
            (model.m3_runtime._decode_current_frame, 'queries'),
            (model.m3_runtime._decode_binding, ('QCSE.Metadata Binding', False, 'QCSE.Total')),
            (propose_fact_answer, ('QCSE.Semantic Routing', False, 'QCSE.Total')),
            (model.m3_runtime.answer_memory.retrieve, ('QCSE.Candidate Retrieval', False, 'QCSE.Total')),
            (model.m3_runtime.answer_memory.construct, ('QCSE.Candidate Construction', False, 'QCSE.Total')),
            (model.m3_runtime.executor.generate, ('QCSE.Target Generation', True, 'QCSE.Total')),
        ]
        if not components:
            specs = [(model.simple_test_pts, 'export_only')]
        try:
            for fn, kind in specs:
                self.patch(fn, kind)
        except BaseException:
            self.close()
            raise

    def patch(self, method, kind):
        fn = inspect.unwrap(getattr(method, '__func__', method))
        if fn.__closure__:
            raise RuntimeError('Cannot safely instrument a closure: ' + fn.__qualname__)
        source = textwrap.dedent(inspect.getsource(fn))
        original = ast.parse(source)
        original.body[0].decorator_list = []
        tree = transform(copy.deepcopy(original), kind)
        recovered = StripScopes().visit(copy.deepcopy(tree))
        if ast.dump(original, include_attributes=False) != ast.dump(recovered, include_attributes=False):
            raise RuntimeError('Instrumentation changed original statements: ' + fn.__qualname__)
        # Defining the same function does not execute its body or change modules.
        namespace = {}
        exec(compile(ast.fix_missing_locations(tree), fn.__code__.co_filename, 'exec'), fn.__globals__, namespace)
        key = id(fn.__globals__)
        if key not in self.globals:
            helpers = dict(__component_span=self.recorder.span,
                __component_defer_export=(self.exports.defer if getattr(self, 'exports', None) is not None
                                          else lambda model, runtime, callback: callback()))
            self.globals[key] = (fn.__globals__, {name: (name in fn.__globals__, fn.__globals__.get(name)) for name in helpers})
            fn.__globals__.update(helpers)
        self.saved.append((fn, fn.__code__))
        fn.__code__ = namespace[fn.__name__].__code__
        self.audit.append(dict(function=fn.__module__ + '.' + fn.__qualname__,
                               source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                               original_statements_preserved=True, boundary=str(kind),
                               output_export_deferred=bool(kind in ('perception', 'export_only')
                                                           and getattr(self, 'exports', None) is not None)))

    def annotate_frame(self, model):
        """Use existing diagnostics, never infer a GPU path from configuration."""
        diag = model.m2_diagnostics
        requests = {r['request_index']: r for r in model.m3_diagnostics['requests']}
        context = self.recorder.context
        for row in self.recorder.rows:
            if row['frame_id'] != context['frame_id']:
                continue
            if row['component_name'] == 'DAVP.Global Execution':
                reused = diag['approx_global_residual_reuse']
                row.update(component_name='DAVP.Global Residual Reuse' if reused else 'DAVP.Global Refresh',
                           execution_path='global_residual_reuse' if reused else 'global_refresh')
            elif row['component_name'] == 'DAVP.Incremental EVA':
                row.update(refreshed_windows=diag['executed_windows'], reused_windows=diag['reused_windows'])
            elif row['component_name'] == 'QCSE.Target Generation':
                request = requests[row['query_id']]
                verified = request.get('verification_forward_calls', 0) > 0
                row.update(component_name='QCSE.Verification and Continuation' if verified else 'QCSE.Native Generation',
                           execution_path='candidate_verification_and_continuation' if verified else 'native_generation')

    def close(self):
        for fn, code in reversed(self.saved):
            fn.__code__ = code
        for namespace, helpers in self.globals.values():
            for name, (existed, value) in helpers.items():
                if existed:
                    namespace[name] = value
                else:
                    namespace.pop(name, None)
        self.saved.clear()
        self.globals.clear()
