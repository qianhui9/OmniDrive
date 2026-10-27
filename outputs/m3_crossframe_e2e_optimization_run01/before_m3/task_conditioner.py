"""Conservative question classification; no ground truth is accepted here."""
import re


def task_type(question):
    q = str(question).lower()
    if re.search(r'\b(safe|safety|risk|collision|happen|planning|trajectory|scenario|environment|conditions|action)\b', q):
        return 'global_safety'
    if re.search(r'\b(lane|road|intersection|turning|topology)\b', q):
        return 'map'
    if re.search(r'\b(how many|distance|far|left|right|front|behind|moving|direction|speed|vehicle|pedestrian|car|truck|bus)\b', q):
        return 'object_relation'
    return 'unknown'


def task_profile(question):
    """Refine the existing classifier for dependency priority, never authorization."""
    task=task_type(question);q=str(question).lower()
    # Clearance, area descriptions and driving decisions require full scene
    # evidence. A vehicle noun alone does not make them object attributes.
    if re.search(r'\b(sufficient space|enough space|pass or park|parking situation|in response to|needs to change lanes|needs to make a turn|should the vehicle|type of area|kind of area)\b',q):
        return task,'full_visual',('object','ego','map')
    if task in ('global_safety','unknown') or re.search(r'\b(how many|count|number|any|none|all|absence|missing|no)\b',q):
        return task,'full_visual',('object','ego','map')
    if task=='map':return task,'map_relations',('map',)
    if re.search(r'\b(speed|velocity|moving|motion|relative|distance|direction)\b',q):
        return task,'object_ego_motion',('object','ego')
    return task,'object_attributes',('object',)
