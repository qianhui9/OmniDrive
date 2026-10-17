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
