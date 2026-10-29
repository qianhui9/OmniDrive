"""Bounded causal answer-token proposals. No tensors, KV, or annotation answers."""
from collections import OrderedDict
from dataclasses import dataclass,replace
from .task_conditioner import question_dependencies,answer_facts
from .world_delta import question_world_delta,relevant,VEHICLES
from time import perf_counter


@dataclass(frozen=True)
class AnswerRecord:
    question: str
    prompt_ids: tuple
    tokens: tuple
    scene_id: str
    frame_id: str
    frame_index: int
    timestamp: float
    world_version: object
    binding: tuple
    source: str = 'current_target_model_generation'
    dependencies: object = None
    world_state: object = None
    polarity: str = 'unknown'
    facts: object = None
    answer_text: str = ''


@dataclass(frozen=True)
class DraftCandidate:
    tokens: tuple
    kind: str
    record: object
    delta: object
    block_length: int = 8
    expected_accepted: float = 2.
    forward_cost_ratio: float = 1.05
    fact_token_start: int = -1
    text: str = ''

    def metadata(self):
        return dict(kind=self.kind,source_frame=self.record.frame_id,
            source_question=self.record.question,tokens=list(self.tokens),text=self.text,
            block_length=self.block_length,expected_accepted=self.expected_accepted,
            forward_cost_ratio=self.forward_cost_ratio,fact_token_start=self.fact_token_start,
            world_delta=self.delta.to_dict() if self.delta else None)


@dataclass(frozen=True)
class DraftSelection:
    record: object = None
    length: int = 0
    match_type: str = 'none'
    exact_available: bool = False
    semantic_available: bool = False
    delta: object = None
    reason: str = 'no_history'
    world_diff_ms: float = 0.
    candidates: tuple = ()
    construction_ms: float = 0.
    tokenizer_ms: float = 0.


class AnswerMemory:
    def __init__(self, max_entries=128, max_age_frames=8, max_age_s=4.):
        self.max_entries = max_entries
        self.max_age_frames = max_age_frames
        self.max_age_s = max_age_s
        self.records = OrderedDict()
        self.acceptance = {}
        self.performance = OrderedDict()

    def reset(self):
        self.records.clear()
        self.acceptance.clear()
        self.performance.clear()

    def valid(self,record,context,binding):
        return (record.scene_id==context['scene_id'] and record.binding==binding
            and 0<context['frame_index']-record.frame_index<=self.max_age_frames
            and 0<context['timestamp']-record.timestamp<=self.max_age_s
            and record.frame_id!=context['frame_id'])

    def select(self, prompt_ids, context, binding):
        key = (context['scene_id'], tuple(prompt_ids))
        record = self.records.get(key)
        if record is None:
            return None
        if not self.valid(record,context,binding):
            return None
        return record

    def retrieve(self,question,prompt_ids,context,binding,current_world=None,guided=False):
        deps=question_dependencies(question)
        exact=self.select(prompt_ids,context,binding)
        if not guided:
            return DraftSelection(exact,8 if exact else 0,'exact' if exact else 'none',
                exact is not None,False,None,'exact_history' if exact else 'no_exact_history')
        def compatible(other):
            if other is None:return False
            if other.signature==deps.signature:return True
            # Positive existential drafts may cross near/all-view wording only
            # when current AND source observations satisfy the CURRENT scope.
            return (other.intent==deps.intent=='existence' and other.classes==deps.classes
                    and other.directions==deps.directions)
        candidates=[r for r in reversed(list(self.records.values())) if self.valid(r,context,binding)
            and compatible(r.dependencies)]
        if exact is not None:candidates=[exact]+[r for r in candidates if r is not exact]
        diff_ms=0.
        for record in candidates:
            tick=perf_counter()
            delta=question_world_delta(record.world_state,current_world,deps)
            diff_ms+=(perf_counter()-tick)*1000
            if (record is exact and current_world is not None and record.world_state is not None
                    and delta.decision_reason not in ('scene_reset','UNKNOWN_state_sequence','UNKNOWN_current_committed_state')):
                # UNKNOWN or geometry change cannot predict target token
                # rejection. Preserve B's complete K8 proposal; causal target
                # acceptance/cost, rather than M1 risk, can later suppress it.
                return DraftSelection(record,8,'exact',True,bool(candidates),delta,
                    'exact_history_preserved_for_target_selection',diff_ms)
            if not delta.suggested_draft_length:continue
            if deps.intent=='existence' and record.polarity!='positive':continue
            key=(record.dependencies.signature,deps.signature)
            outcome=self.acceptance.get(key)
            if outcome and outcome[0]==0 and context['frame_index']-outcome[1]<=2:continue
            length=delta.suggested_draft_length
            return DraftSelection(record,length,'exact' if record is exact else 'semantic',
                exact is not None,bool(candidates),delta,delta.decision_reason,diff_ms)
        tick=perf_counter()
        delta=question_world_delta(candidates[0].world_state if candidates else None,current_world,deps)
        diff_ms+=(perf_counter()-tick)*1000
        return DraftSelection(None,0,'none',exact is not None,bool(candidates),delta,
            delta.decision_reason if candidates else 'no_compatible_semantic_history',diff_ms)

    def construct(self,selection,question,prompt_ids,context,binding,current_world,tokenizer):
        """At most four deterministic proposals; no forward, GT or current answer.

        M1 controls which historical *facts* may seed composition. It does not
        certify these strings as current answers. Every candidate still enters
        the one current-target verifier, after prefill-aware selection.
        """
        tick=perf_counter();deps=question_dependencies(question);out=[];token_ms=0.;diff_ms=selection.world_diff_ms
        def add(tokens,kind,record,delta,text='',fact_start=-1):
            tokens=tuple(tokens)
            if len(tokens)<2 or any(c.tokens==tokens for c in out) or len(out)>=4:return
            stat=self.performance.get((record.prompt_ids if kind=='exact' else deps.signature,kind))
            # Two consecutive unproductive requests, not an UNKNOWN world flag.
            if stat and stat['unproductive_streak']>=2:return
            expected=stat['accepted_per_verification'] if stat else (6. if kind=='exact' else 2.)
            out.append(DraftCandidate(tokens,kind,record,delta,8,min(8.,max(1.,expected)),
                stat['forward_cost_ratio'] if stat else 1.05,fact_start,text))
        if selection.record is not None:
            kind='exact' if selection.exact_available and selection.match_type=='exact' else 'semantic_raw'
            add(selection.record.tokens,kind,selection.record,selection.delta,selection.record.answer_text)
            if kind=='exact':
                return replace(selection,candidates=tuple(out),construction_ms=(perf_counter()-tick)*1000)
        if current_world is not None and not deps.complete_coverage_required and deps.intent in ('category','existence','position','motion'):
            wanted=set(deps.classes)
            if 'vehicle' in wanted:wanted=(wanted-{'vehicle'})|VEHICLES
            # Candidate ordering uses prior facts, recent source and M1 support,
            # never the current generated answer. Only two source records need
            # expensive delta/tokenizer work.
            source_count=0
            for record in reversed(list(self.records.values())):
                if len(out)>=4 or source_count>=2:break
                if not self.valid(record,context,binding) or record.world_state is None:continue
                f=record.facts
                if f is None or f.polarity!='positive' or f.uncertain or not f.classes:continue
                classes=set(f.classes)&wanted if wanted else set(f.classes)
                if not classes:continue
                t=perf_counter();delta=question_world_delta(record.world_state,current_world,deps);diff_ms+=(perf_counter()-t)*1000
                if delta.decision_reason in ('scene_reset','UNKNOWN_state_sequence'):continue
                previous={a.semantic_class for a in record.world_state.anchors if a.observed and relevant(a,deps)}
                now={a.semantic_class for a in current_world.anchors if a.observed and relevant(a,deps)}
                supported=classes&previous&now
                if not supported:continue
                source_count+=1
                # Original semantic answer remains one candidate, but a suffix
                # sliced from its middle is never presented as an answer start.
                if record.dependencies.intent==deps.intent:
                    add(record.tokens,'semantic_raw',record,delta,record.answer_text)
                if tokenizer is None:continue
                noun=sorted(supported)[0].replace('_',' ')
                article='an' if noun[:1] in 'aeiou' else 'a'
                # Reuse direction only if historical content AND current/source
                # observed scope agree; UNKNOWN identity remains on the delta.
                direction=next((d for d in deps.directions if d in f.directions),None)
                location={'front':'in front','rear':'behind','left':'on the left','right':'on the right'}.get(direction,'')
                if deps.intent=='existence':texts=[f'Yes, there is {article} {noun}.',f'There is {article} {noun}.']
                elif deps.intent=='category':texts=[f'The vehicle{(" "+location) if location else ""} is {article} {noun}.',f'There is {article} {noun}.']
                else:texts=[f'The {noun}']  # class-only prefix: no stale motion/position assertion
                for text in texts:
                    t=perf_counter();tokens=tokenizer.encode(text,add_special_tokens=False)
                    # Prefix encoding gives a conservative fact-start boundary
                    # under SentencePiece; no unrelated mid-answer token slice.
                    start=max(0,len(tokenizer.encode(text[:text.index(noun)],add_special_tokens=False))-1)
                    token_ms+=(perf_counter()-t)*1000
                    add(tokens,'question_conditioned',record,delta,text,start)
                    if len(out)>=4:break
        return replace(selection,candidates=tuple(out),world_diff_ms=diff_ms,
            semantic_available=selection.semantic_available or any(c.kind!='exact' for c in out),
            construction_ms=(perf_counter()-tick)*1000-(diff_ms-selection.world_diff_ms)-token_ms,tokenizer_ms=token_ms)

    def observe_acceptance(self,selection,diagnostics,frame_index):
        if selection.record is None:return
        logs=diagnostics.get('verification_log',[])
        if logs:
            signature=selection.record.dependencies.signature
            target_signature=selection.delta.question_signature if selection.delta else signature
            self.acceptance[(signature,target_signature)]=(logs[0]['accepted'],frame_index)
            if not diagnostics.get('target_aware_enabled'):return
            kind=diagnostics.get('selected_candidate_kind',selection.match_type)
            key=(selection.record.prompt_ids if kind=='exact' else target_signature,kind)
            count=len(logs);accepted=sum(l['accepted'] for l in logs)
            tokens=len(diagnostics.get('output_token_ids',[]))
            calls=diagnostics.get('decode_forward_calls',0)
            step=diagnostics.get('decode_forward_cuda_ms',0.)/max(1,calls)
            saved=tokens-1-calls
            old=self.performance.get(key,{})
            self.performance[key]=dict(requests=old.get('requests',0)+1,recent_accepted=accepted,
                accepted_per_verification=accepted/count,
                acceptance_rate=accepted/max(1,sum(len(l['proposed']) for l in logs)),
                first_rejection=logs[0]['rejection_position'],mean_verification_ms=step,
                forward_cost_ratio=1.05,estimated_saved_forwards=saved,
                estimated_net_ms=saved*step-diagnostics.get('draft_selection_wall_ms',0.),
                unproductive_streak=old.get('unproductive_streak',0)+1 if accepted<2 or saved<=0 else 0)
            self.performance.move_to_end(key)
            while len(self.performance)>self.max_entries:self.performance.popitem(last=False)

    def remember(self, question, prompt_ids, tokens, context, binding,world_state=None,answer_text='',extract_facts=True):
        ids = tuple(prompt_ids)
        key = (context['scene_id'], ids)
        facts=answer_facts(answer_text) if extract_facts else None
        self.records[key] = AnswerRecord(str(question), ids, tuple(tokens), context['scene_id'],
            context['frame_id'], context['frame_index'], context['timestamp'],
            context.get('current_state_version',context.get('state_version')), binding,
            dependencies=question_dependencies(question),world_state=world_state,
            polarity=facts.polarity if facts else 'unknown',facts=facts,answer_text=str(answer_text))
        self.records.move_to_end(key)
        while len(self.records) > self.max_entries:
            self.records.popitem(last=False)
