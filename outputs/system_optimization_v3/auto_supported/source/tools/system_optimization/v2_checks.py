"""Strict paired quality gates for second-round executions, never timing gates."""
import json
from pathlib import Path
from .audit import write


def rows(path):
    return [json.loads(line) for line in Path(path).open()]


def check(path, required=True):
    path=Path(path);records=rows(path/'frames.jsonl')
    references={(r['mode'],r['repeat'],r['index']):r for r in records if r['mode'] in ('m1','m12')}
    pairs=[]
    for row in records:
        mapping=dict(opt_serial='m1',overlap='m1',unified_auto='m1',opt_m12='m12')
        if row['mode'] not in mapping:continue
        key=(mapping[row['mode']],row['repeat'],row['index'])
        if key not in references:continue
        ref=references[key]
        quality=row['frame']['quality']
        numerical=quality['numerical']
        # Typed feature/memory equality also rejects dtype or shape changes.
        exact_numeric=quality['exact_perception_memory_digest'] and all(
            v.get('max_abs')==0 and not v.get('shape_mismatches') for v in numerical.values())
        pair=dict(mode=row['mode'],reference=ref['mode'],index=row['index'],repeat=row['repeat'],
            identity=row['dataset_identity'],m1_state_equal=quality['m1_state_digest']==ref['frame']['quality']['m1_state_digest'],
            complete_token_equality=row['generated_ids']==ref['generated_ids'],
            perception_feature_memory_equal=bool(exact_numeric),native_token_equal=quality['exact_token_equality'])
        pair['pass']=all(pair[k] for k in ('m1_state_equal','complete_token_equality','perception_feature_memory_equal','native_token_equal'))
        pairs.append(pair)
    result=dict(source=str(path),pairs=pairs,paired_frames=len(pairs),
        passed=bool(pairs) and all(p['pass'] for p in pairs),
        timing_fields_excluded_from_state_digest=True)
    write(path/'overlap_correctness.json',result)
    if required and not result['passed']:raise AssertionError('Strict overlap quality gate failed; see '+str(path/'overlap_correctness.json'))
    return result
