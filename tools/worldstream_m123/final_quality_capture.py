"""External return-value feature observer for the final quality audit.

All inference is delegated to critical_path and the sole serving executor.
Feature comparisons run after the synchronized model-wall timer has stopped.
"""
import argparse
from dataclasses import asdict
from pathlib import Path

from . import critical_path as worker

ROOT = Path('outputs/m123_final_quality_validation_run01')


def feature_error(actual, reference):
    """NCHW diagnostics, including spatial tiles; never a quality label."""
    import torch
    x, ref = actual.float(), reference.float()
    delta = x - ref

    def stats(d, r):
        return dict(relative_l2=float(d.norm() / r.norm().clamp_min(1e-12)),
                    rmse=float(d.square().mean().sqrt()), max_abs=float(d.abs().max()))

    result = stats(delta, ref)
    result['shape'] = list(x.shape)
    result['per_camera'] = [stats(d, r) for d, r in zip(delta, ref)]
    result['tiles_4x4'] = [dict(camera=n, row=i, col=j, **stats(d, r))
        for n in range(x.shape[0])
        for i, (ds, rs) in enumerate(zip(torch.tensor_split(delta[n], 4, -2),
                                        torch.tensor_split(ref[n], 4, -2)))
        for j, (d, r) in enumerate(zip(torch.tensor_split(ds, 4, -1),
                                     torch.tensor_split(rs, 4, -1)))]
    token_error = delta.square().sum(1).sqrt() / ref.square().sum(1).sqrt().clamp_min(1e-12)
    result['spatial_token_relative_l2_quantiles'] = {
        str(q): float(torch.quantile(token_error.flatten(), q)) for q in (.5, .9, .99)}
    return result


def install():
    original_receipt = worker.visual_receipt
    original_configure = worker.configure_job
    original_sources = worker.execution_sources

    def sources():
        return {p: h for p, h in original_sources().items()
                if not p.endswith('final_quality_analysis.py')}

    def configure(model, job, root, observer):
        original_configure(model, job, root, observer)
        worker.append(root / job['audit_stage'] / 'effective_runtime_jobs.jsonl',
                      dict(label=job['label'], scene=job['scene']['scene_id'],
                           M1=asdict(model.m1_manager.config), M2=asdict(model._continuous_m2.config)))

    def receipt(model, outputs, holder, job, fi, root):
        import torch
        value = original_receipt(model, outputs, holder, job, fi, root)
        value['feature_observation'] = 'actual_forward_from_m1_return_NCHW'
        if job.get('audit_map_capture'):
            lanes, scores = outputs[0]['lane_results'][0]
            value['lane_predictions'] = dict(points=[p.tolist() for p in lanes], scores=scores.tolist())
        key = job.get('feature_reference_key')
        if key:
            folder = Path('/tmp/omnidrive-final-quality-features') / key
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (str(fi) + '.pt')
            x = holder['tensor'].cpu()
            if job.get('audit_save_features'):
                torch.save(x, path)
            if job.get('audit_compare_features'):
                value['full_feature_difference'] = feature_error(x, torch.load(path, map_location='cpu'))
        return value

    worker.configure_job, worker.visual_receipt, worker.execution_sources = configure, receipt, sources


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    install()
    with (args.root / (args.stage + '_execution.log')).open('x') as log:
        from contextlib import redirect_stdout, redirect_stderr
        import traceback
        with redirect_stdout(log), redirect_stderr(log):
            try:
                worker.run(args.root, args.stage)
            except BaseException:
                worker.write(args.root / (args.stage + '_failure.json'), dict(error=traceback.format_exc()))
                raise


if __name__ == '__main__':
    main()
