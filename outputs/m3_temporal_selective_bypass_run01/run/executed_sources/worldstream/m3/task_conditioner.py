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
    elif re.search(r'\b(lanes?|roads?|intersection|junction|topology)\b',q) and (
            not classes or re.search(r'\b(?:type|kind|category) of (?:road|lane)\b',q)):
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
    negative_query=bool(re.search(r'\b(no|none|not|without|absence)\b',q))
    if negative_query and 'complete_scene' not in attrs:attrs+=('complete_scene',)
    return QuestionDependencies(intent,classes,directions,attrs,signature,
        negative_query or intent in ('count','signal','safety','environment','unknown'))


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


@dataclass(frozen=True)
class QuestionFacts:
    """Required answer facts for the explicit, limited WRSAR language domain."""
    intent: str = 'unsupported'
    classes: tuple = ()
    directions: tuple = ()
    near_m: float = 50.
    direct: bool = False
    selector: str = 'any'
    required: tuple = ()
    reason: str = 'outside_structured_object_domain'

    @property
    def signature(self):
        return (self.intent,self.classes,self.directions,self.near_m,self.direct,self.selector)


def question_facts(question):
    """No GT, history answer, or current answer participates in parsing.

    Plural categories, disjunctions, multi-part instructions, road roles and
    global decisions cannot be answered by one positive object witness.
    """
    q=' '.join(str(question).lower().strip().rstrip('?.').split())
    dep=question_dependencies(q)
    if re.search(r'\b(and|or|all|every|no|not|count|many|color|colour|lane|road|crosswalk|crossing|traffic|safe|safety|should|weather|construction|oncoming|following)\b',q):
        # "moving or stationary" is the sole structured disjunction.
        if not re.fullmatch(r'is the nearest (?:car|vehicle) moving or stationary',q):return QuestionFacts()
    classes=dep.classes
    near=30. if re.search(r'\b(near|nearby|close|vicinity|immediate)\b',q) else 50.
    direct=bool(re.search(r'\bdirectly\b',q))
    if re.fullmatch(r'what (?:type|kind) of (?:vehicle|object) is (?:directly )?(?:ahead|in front|on the right|on the left)(?: of (?:us|the vehicle))?',q):
        dirs=dep.directions or ('front',)
        return QuestionFacts('category',('object',) if 'of object' in q else ('vehicle',),dirs,near,direct,'nearest',
                             ('target_class','target_region','current_observation'),'structured_category')
    if (len(classes)==1 and re.fullmatch(r'(?:is|are) there (?:any |a |an )?(?:cars?|vehicles?|pedestrians?|people|trucks?|buses?|bicycles?|motorcycles?)(?: (?:visible |present )?(?:in front|ahead|on the left|on the right|behind|nearby|near the vehicle|close to the vehicle|in the immediate vicinity))?(?: of (?:us|the vehicle))?',q)):
        return QuestionFacts('existence',classes,dep.directions,near,direct,'any',
                             ('positive_existence','target_region','current_observation'),'positive_witness_only')
    match=re.fullmatch(r'where is the nearest (car|vehicle|pedestrian) relative to (?:the vehicle|ego|us)',q)
    if match:
        return QuestionFacts('position',(match.group(1),),(),near,False,'nearest',
                             ('target_class','longitudinal_direction','lateral_direction','current_observation'),'structured_position')
    if re.fullmatch(r'is the nearest (?:car|vehicle) moving or stationary',q):
        return QuestionFacts('motion',classes,(),near,False,'nearest',
                             ('target_class','reliable_velocity','current_observation'),'requires_reliable_velocity')
    return QuestionFacts()
