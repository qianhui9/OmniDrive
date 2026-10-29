"""Conservative question classification; no ground truth is accepted here."""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionDependencies:
    intent: str
    classes: tuple = ()
    directions: tuple = ()
    attributes: tuple = ()
    signature: tuple = ()
    complete_coverage_required: bool = False


@dataclass(frozen=True)
class AnswerFacts:
    polarity: str = 'unknown'
    classes: tuple = ()
    directions: tuple = ()
    motion: str = 'unknown'
    uncertain: bool = False


def answer_facts(answer):
    """Conservative statements from OUR earlier output, never current truth.

    Negated, modal and interrogative clauses cannot create positive facts.
    Multi-object directions are deliberately left unbound rather than joining
    the direction of one object to another object's class.
    """
    text=' '.join(str(answer).lower().split())
    classes=set();directions=set();positive=False;negative=False;uncertain=False;motions=set()
    for clause,ending in re.findall(r'([^.!?;]+)([.!?;]?)',text):
        if not clause.strip():continue
        found=question_dependencies('What type of '+clause+'?').classes
        modal=bool(re.search(r'\b(may|might|possibly|perhaps|uncertain|unsure|cannot determine|could be|appears? to|seems? to)\b',clause))
        question=ending=='?' or bool(re.match(r'\s*(?:is there|are there|what|which|where|whether|can you|do you)',clause))
        negated=bool(re.search(r"\b(no|not|none|without|neither|cannot|can't|isn't|aren't|doesn't|don't)\b",clause))
        uncertain|=modal or question
        if negated:negative=True;continue
        if modal or question:continue
        assertion=bool(re.search(r'\b(yes|there (?:is|are)|visible|seen|see|present|ahead|front|left|right|behind|parked|moving|stationary|stopped)\b',clause))
        if found and assertion:
            positive=True;classes.update(found)
            if len(found)==1:
                for d,pattern in [('front','front|ahead'),('rear','behind|rear'),('left','left'),('right','right')]:
                    if re.search(r'\b(?:'+pattern+r')\b',clause):directions.add(d)
            if re.search(r'\b(parked|stationary|stopped)\b',clause):motions.add('stationary')
            if re.search(r'\b(moving|driving|traveling|travelling)\b',clause):motions.add('moving')
    polarity='positive' if positive and not negative else 'negative' if negative and not positive else 'unknown'
    return AnswerFacts(polarity,tuple(sorted(classes)),tuple(sorted(directions)),
                       next(iter(motions)) if len(motions)==1 else 'unknown',uncertain)


def question_dependencies(question):
    """One deterministic parser for proposal retrieval, never answer authority."""
    q=' '.join(str(question).lower().split())
    entities=re.sub(r'\b(?:your|our|ego|this)\s+(?:vehicle|car)\b','ego',q)
    entities=re.sub(r'\b(?:near|from|around) the vehicle(?:\x27s current position)?\b','ego',entities)
    classes=tuple(c for c,pattern in (
        ('pedestrian',r'pedestrians?|people|person'),('car',r'cars?'),('truck',r'trucks?'),
        ('bus',r'buses|bus'),('bicycle',r'bicycles?|cyclists?|bikes?'),
        ('motorcycle',r'motorcycles?'),('vehicle',r'vehicles?|traffic participants?'))
        if re.search(r'\b(?:'+pattern+r')\b',entities))
    directions=tuple(d for d,pattern in (
        ('front',r'front|ahead|forward'),('rear',r'behind|rear|back'),
        ('left',r'left'),('right',r'right')) if re.search(r'\b(?:'+pattern+r')\b',q))
    if re.search(r'\b(safe|safety|risk|collision|should|recommend|happen|action|planning|trajectory|clear path|obstructed|blocked path)\b',q):
        intent='safety';attrs=('complete_scene','ego','motion')
    elif re.search(r'\b(traffic (?:light|signal|sign)|signals?|signs?)\b',q):
        intent='signal';attrs=('signal_state','complete_scene')
    elif re.search(r'\b(pedestrian crossing|crosswalk|bus stops?)\b',q):
        intent='road';attrs=('map_geometry','map_class','ego')
    elif re.search(r'\b(weather|environment|conditions|buildings?|establishments?|business|commercial|setting|area|overview|surroundings)\b',q):
        intent='environment';attrs=('scene_appearance','complete_scene')
    elif re.search(r'\b(lanes?|roads?|intersection|junction|topology)\b',q) and not classes:
        intent='road';attrs=('map_geometry','map_class','ego')
    elif re.search(r'\b(how many|count|number of)\b',q):
        intent='count';attrs=('class','existence','complete_scene')
    elif re.search(r'\b(distance|far|close|nearby|proximity|vicinity)\b',q) and not re.search(r'\b(is there|are there|any)\b',q):
        intent='distance';attrs=('class','identity','geometry','ego')
    elif re.search(r'\b(speed|velocity|moving|motion|stationary|parked|direction|travel)\b',q):
        intent='motion';attrs=('class','identity','velocity','ego')
    elif classes and re.search(r'\b(type|types|kind|category|class)\b',q):
        intent='category';attrs=('class','identity')+(('geometry','ego') if directions else ())
    elif re.search(r'\b(where|position|located|relative)\b',q) or directions:
        intent='position';attrs=('class','identity','geometry','ego')
    elif classes and re.search(r'\b(is there|are there|any|visible|see|present|observe|other vehicles)\b',q):
        intent='existence';attrs=('class','positive_observation')
    else:
        intent='unknown';attrs=('unknown',)
    # A nearby existence question depends on geometry as well as positivity.
    if intent=='existence' and re.search(r'\b(immediate|nearby|vicinity|proximity|close)\b',q):
        attrs+=('near',)
    scope='near' if 'near' in attrs else 'all_views'
    signature=(intent,classes,directions,scope)
    return QuestionDependencies(intent,classes,directions,attrs,signature,
        intent in ('count','signal','safety','environment','unknown'))


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
