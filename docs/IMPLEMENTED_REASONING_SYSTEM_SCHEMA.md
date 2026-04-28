# Implemented Reasoning System Schema

Portable reference for the current Irys RLM implementation.

Date captured: 2026-04-23

Current schema version: 58

This file is intentionally standalone. It is written so it can be copied out of this repository and used as the basis for discussing a general-purpose ontology mapping and reasoning system. It describes what is implemented now, not only what is intended.

## 1. System In One Sentence

The system is a durable reasoning layer that turns source material into reusable assertions, links assertions into issue and proof graphs, tracks provenance and verification, propagates conflicts through dependent claims, and uses that stored state to avoid rediscovering facts on later runs.

In the current repository this is legal-flavored, but most of the implemented machinery is more general:

- A workspace has source artifacts.
- Source artifacts contain spans.
- Spans produce claim occurrences.
- Claim occurrences canonicalize into reusable claims.
- Claims connect through support, attack, contradiction, dependency, and supersession links.
- Claims and other objects support or attack objective nodes.
- Objective nodes have criteria.
- Criteria have proof coverage.
- Review state, provenance, trust, and policy decide which objects can be used for which purpose.
- Runs append an audit ledger and reuse the existing graph.

## 2. What Is Legal-Specific Versus General

The current implementation has legal names, legal enum values, and legal templates. The underlying shape is broader.

### General-purpose substrate already implemented

These concepts are not inherently legal:

- `matter`: a workspace or reasoning context.
- `document_inventory`: a known source artifact.
- `document_card`: profile metadata for a source artifact.
- `span`: a stable source range inside an artifact.
- `actor`: an entity, person, organization, or participant.
- `assertion`: a canonical claim.
- `assertion_occurrence`: one source mention or extraction of a canonical claim.
- `assertion_link`: graph relation between claims.
- `evidence_edge`: support or attack edge from a source object to a target object.
- `issue`: an objective node, question, topic, concept, task, or ontology node.
- `issue_predicate`: a criterion, element, required slot, or testable property.
- `gap`: missing input, missing evidence, unresolved contradiction, or missing metadata.
- `verification_state`: review lifecycle for generated or imported objects.
- `provenance_event`: derivation metadata.
- `belief_revision_event`: truth-maintenance audit.
- `run_session` and `ledger_event`: durable run history and resumability.
- `reasoning_cache`: scoped plan and route cache.
- `content_policy_audit`: visibility and eligibility decisions.

### Legal overlays that would be renamed or replaced for ontology mapping

These are currently legal-flavored:

- `SourceRole` values such as `advocacy`, `operative`, `procedural`, and `authoritative`.
- `SpeechAct` values such as `alleged`, `admitted`, `ordered`, `testified`, and `waived`.
- `IssueType` values such as `claim`, `defense`, `contract_question`, and `damages`.
- Built-in issue templates such as `contract_breach` and `negligence`.
- `authority` and `authority_issue_link`, which assume legal authority and precedent.
- `privilege_flag`, clean mode, and attorney review semantics.
- Deliverable routes such as privilege logs.
- The `legal` value in `model_layer`.

For a general ontology mapper, these should become profile-defined vocabularies and templates over a stable reasoning kernel, rather than hardcoded legal enums.

## 3. Main Runtime Flow

The public entry point is `Irys.investigate()` in `src/irys/api.py`.

High-level flow:

1. Validate the query and repository.
2. Initialize the LLM client and RLM engine.
3. Open a `MatterModel` for the repository when matter mode is enabled.
4. Handle simple social niceties with a fast NANO model route.
5. Build an answerability snapshot from the matter store.
6. Use `CascadeGovernor` to classify the route.
7. Execute one route family:
   - `query`: answer from durable state when possible.
   - `read`: synthesize from matter state and source context.
   - `investigate`: run the full recursive lead-mining engine.
   - `trace`: explain previous reasoning or provenance.
   - `compare`: compare current and previous run state.
   - `scenario`: apply a temporary assumption and reason against it.
   - `steer`: parse a requested mutation and preview candidates.
   - `clarify`: ask a targeted question.
   - `deliverable`: render a supported work product.
8. Append route and run decisions into the ledger.
9. During investigation, extracted facts are recorded into the matter graph.
10. At the end of the run, pending belief propagation is flushed and proof states can be recomputed.

The important property is that every run starts from durable state. The system does not treat each prompt as a blank slate.

## 4. MatterModel Facade

`src/irys/matter/matter.py` contains the central `MatterModel` facade.

There is one SQLite matter database per repository, stored at:

```text
<repository>/.irys/matter.sqlite3
```

The facade wires together the implemented stores:

```text
MatterModel
  AssertionStore
  GapStore
  ActorStore
  IssueStore
  ClarificationStore
  QuantStore
  ReasoningLedgerStore
  BeliefRevisionEngine
  DocumentInventoryStore
  DocumentCardStore
  SpanStore
  DocumentActorRoleStore
  ReasoningCacheStore
  TrustOverrideStore
  DocumentAnnotationStore
  DecisionContextStore
  AuthorityStore
  ProofStateStore
  AssumptionStore
  VerificationStateStore
  EvidenceStore
  PrivilegeGate
  ProvenanceStore
  ContentPolicyGuard
```

### Durable database behavior

The schema uses:

- SQLite.
- WAL mode.
- `foreign_keys=ON`.
- `STRICT` tables.
- JSON1 patterns for JSON text columns.
- FTS5 where needed by the wider project.
- migration tracking via `schema_migration` and `schema_version`.

### Durable propagation queues

The `pending_propagation` table stores unfinished correction or evidence propagation work. If a belief revision pass hits a work budget or conflict retry limit, remaining assertion ids can be persisted and drained later.

This is important because conflict propagation is graph-shaped and can exceed a single request budget.

## 5. Exact Current SQLite Schema

This section lists the live tables and columns after applying migrations for schema version 58.

### `actor`

Purpose: canonical entity/participant records.

```text
id TEXT PK
matter_id TEXT
canonical_name TEXT
normalized_name TEXT
actor_type TEXT
home_side TEXT
agenda_notes TEXT
reliability_notes TEXT
created_at TEXT
updated_at TEXT
```

### `actor_affiliation`

Purpose: relationship between an actor and an organization actor.

```text
id TEXT PK
actor_id TEXT
org_actor_id TEXT
role TEXT
start_date TEXT
end_date TEXT
created_at TEXT
```

### `actor_alias`

Purpose: alternate names for actor resolution.

```text
id TEXT PK
actor_id TEXT
alias_text TEXT
alias_type TEXT
created_at TEXT
```

### `assertion`

Purpose: canonical reusable claim.

```text
id TEXT PK
matter_id TEXT
proposition_key TEXT
claim_key TEXT
identity_version TEXT
proposition_text TEXT
model_layer TEXT
assertion_kind TEXT
polarity TEXT
canonical_subject_key TEXT
subject_ref_type TEXT
subject_ref_id TEXT
predicate_key TEXT
canonical_object_key TEXT
object_json TEXT
temporal_scope_start TEXT
temporal_scope_end TEXT
temporal_identity_key TEXT
speaker_scope_key TEXT
canonicalization_confidence REAL
belief_state TEXT
confidence REAL
created_at TEXT
updated_at TEXT
```

Key invariant: there is one canonical assertion per `(matter_id, model_layer, claim_key)`.

### `assertion_issue_link`

Purpose: legacy direct relation between assertions and issues.

```text
id TEXT PK
assertion_id TEXT
issue_id TEXT
relation_type TEXT
created_at TEXT
```

### `assertion_link`

Purpose: relation between two canonical assertions.

```text
id TEXT PK
src_assertion_id TEXT
dst_assertion_id TEXT
link_type TEXT
weight REAL
created_at TEXT
```

Common link types: `supports`, `attacks`, `depends_on`, `supersedes`, `contradicts`, `corroborates`.

### `assertion_occurrence`

Purpose: one occurrence, mention, extraction, or source instance of a canonical assertion.

```text
id TEXT PK
assertion_id TEXT
document_id TEXT
document_inventory_id TEXT
doc_basename TEXT
raw_text TEXT
span_id TEXT
speaker_actor_id TEXT
source_role TEXT
source_side TEXT
speech_act TEXT
origin_kind TEXT
subject_ref_type TEXT
subject_ref_id TEXT
predicate_key TEXT
object_json TEXT
temporal_scope_start TEXT
temporal_scope_end TEXT
polarity TEXT
speaker_scope_key TEXT
claim_key_candidate TEXT
resolution_strategy TEXT
extraction_confidence REAL
created_at TEXT
```

Key invariant: many occurrences can point to one canonical assertion. This is the main fact reuse mechanism.

### `assertion_revision`

Purpose: field-level audit trail for assertion changes.

```text
id TEXT PK
batch_id TEXT
assertion_id TEXT
changed_field TEXT
old_value_json TEXT
new_value_json TEXT
actor_kind TEXT
actor_ref TEXT
cause TEXT
run_id TEXT
note TEXT
created_at TEXT
```

### `assumption`

Purpose: explicit provisional or user-provided assumption.

```text
id TEXT PK
matter_id TEXT
statement TEXT
rationale TEXT
invalidation_condition TEXT
source_kind TEXT
status TEXT
created_at TEXT
updated_at TEXT
```

### `assumption_link`

Purpose: attach assumptions to target objects.

```text
id TEXT PK
assumption_id TEXT
target_type TEXT
target_id TEXT
created_at TEXT
```

### `authority`

Purpose: legal authority or external rule source.

```text
id TEXT PK
matter_id TEXT
authority_type TEXT
citation TEXT
name TEXT
jurisdiction TEXT
decided_at TEXT
holdings TEXT
key_rules TEXT
weight TEXT
precedential_rank INTEGER
applicability TEXT
source_doc_id TEXT
source_span_id TEXT
created_at TEXT
updated_at TEXT
```

For general ontology mapping, this would become a reference source, standard, canonical vocabulary entry, ontology document, or rule source.

### `authority_issue_link`

Purpose: connect authorities to issues.

```text
authority_id TEXT PK
issue_id TEXT PK
relevance TEXT
created_at TEXT
```

### `belief_revision_event`

Purpose: audit row for belief state and confidence changes.

```text
id TEXT PK
assertion_id TEXT
run_id TEXT
cause TEXT
old_belief_state TEXT
new_belief_state TEXT
old_confidence REAL
new_confidence REAL
note TEXT
created_at TEXT
```

### `clarification_question`

Purpose: durable question generated from a gap.

```text
id TEXT PK
matter_id TEXT
gap_id TEXT
run_id TEXT
question_text TEXT
why_it_matters TEXT
expected_impact TEXT
answer_text TEXT
answered_at TEXT
status TEXT
created_at TEXT
```

### `content_policy_audit`

Purpose: audit of visibility, trust, and clean-mode policy decisions.

```text
id TEXT PK
matter_id TEXT
purpose TEXT
policy_audience TEXT
target_kind TEXT
target_id TEXT
action TEXT
reason_code TEXT
trust_bucket TEXT
privilege_flag INTEGER
note TEXT
created_at TEXT
```

### `decision_context`

Purpose: explicit objective and decision-maker context.

```text
id TEXT PK
matter_id TEXT
decision_maker_type TEXT
decision_maker_name TEXT
objective TEXT
strategic_notes TEXT
scope_narrow INTEGER
created_at TEXT
updated_at TEXT
```

### `document_actor_role`

Purpose: relation between a document and an actor.

```text
id TEXT PK
doc_id TEXT
actor_id TEXT
role_type TEXT
raw_name TEXT
confidence REAL
created_at TEXT
```

### `document_annotation`

Purpose: user/system note attached to document patterns.

```text
id TEXT PK
matter_id TEXT
document_pattern TEXT
annotation_text TEXT
annotation_type TEXT
created_at TEXT
updated_at TEXT
```

### `document_card`

Purpose: profiled metadata for a source artifact.

```text
id TEXT PK
doc_id TEXT
title TEXT
doc_type TEXT
doc_subtype TEXT
source_side TEXT
author TEXT
sender TEXT
recipient TEXT
creation_date TEXT
sent_date TEXT
effective_date TEXT
discovery_date TEXT
purpose TEXT
rhetorical_posture TEXT
reliability_posture TEXT
operative_status TEXT
privilege_flag INTEGER
unresolved_flags TEXT
source_role TEXT
signatories_json TEXT
created_at TEXT
updated_at TEXT
```

### `document_inventory`

Purpose: indexed source artifact inventory.

```text
id TEXT PK
matter_id TEXT
relative_path TEXT
storage_uri TEXT
sha256 TEXT
size_bytes INTEGER
file_type TEXT
modified_at TEXT
discovered_at TEXT
ingest_status TEXT
parse_status TEXT
duplicate_cluster TEXT
family_id TEXT
version_chain_id TEXT
salience_score REAL
last_read_at TEXT
maintenance_status TEXT
profiled_at TEXT
last_maintained_at TEXT
```

### `document_relation`

Purpose: relation between source artifacts, such as version or attachment relationships.

```text
id TEXT PK
source_doc_id TEXT
target_doc_id TEXT
relation_type TEXT
confidence REAL
created_at TEXT
```

### `document_trust_override`

Purpose: explicit trust override for matching document patterns.

```text
id TEXT PK
matter_id TEXT
document_pattern TEXT
trust_level TEXT
note TEXT
created_at TEXT
```

### `evidence_edge`

Purpose: canonical support or attack edge from a source object to a target object.

```text
id TEXT PK
matter_id TEXT
source_kind TEXT
source_id TEXT
source_document_inventory_id TEXT
source_span_id TEXT
source_occurrence_id TEXT
target_kind TEXT
target_id TEXT
relation_type TEXT
proof_weight REAL
source_confidence REAL
admissibility_status TEXT
vulnerability_json TEXT
note TEXT
verification_status TEXT
independence_factor REAL
backfill_source TEXT
source_identity_status TEXT
origin_kind TEXT
active INTEGER
effective_weight REAL
independence_cluster_id TEXT
created_at TEXT
updated_at TEXT
```

Key invariant: the natural key is `(matter_id, source_kind, source_id, target_kind, target_id, relation_type)`.

### `evidence_link`

Purpose: older proof-link table retained for compatibility.

```text
id TEXT PK
target_type TEXT
target_id TEXT
source_type TEXT
source_id TEXT
relation_type TEXT
proof_weight REAL
source_diversity INTEGER
auth_status TEXT
admissibility_status TEXT
vulnerability_json TEXT
note TEXT
created_at TEXT
```

The newer production substrate is `evidence_edge`.

### `gap`

Purpose: missingness, blocker, missing artifact, missing metadata, or unresolved contradiction.

```text
id TEXT PK
matter_id TEXT
gap_type TEXT
description TEXT
expected_artifact TEXT
materiality_score REAL
blocker_score REAL
status TEXT
resolution_note TEXT
created_at TEXT
updated_at TEXT
description_key TEXT
```

### `gap_link`

Purpose: attach gaps to affected objects.

```text
id TEXT PK
gap_id TEXT
affected_type TEXT
affected_id TEXT
created_at TEXT
```

### `issue`

Purpose: objective node, issue, concept, question, or reasoning target.

```text
id TEXT PK
matter_id TEXT
parent_issue_id TEXT
title TEXT
issue_type TEXT
burden_side TEXT
materiality REAL
salience REAL
status TEXT
sort_order INTEGER
created_at TEXT
updated_at TEXT
```

### `issue_predicate`

Purpose: testable criterion, element, slot, or requirement under an issue.

```text
id TEXT PK
issue_id TEXT
description TEXT
burden_side TEXT
status TEXT
created_at TEXT
template_id TEXT
template_version TEXT
element_key TEXT
element_order INTEGER
```

### `ledger_event`

Purpose: ordered event log for a run.

```text
id TEXT PK
run_id TEXT
seq_no INTEGER
event_type TEXT
why TEXT
summary TEXT
branch_issue_id TEXT
changed_object_type TEXT
changed_object_id TEXT
snapshot_json TEXT
created_at TEXT
```

### `llm_call`

Purpose: durable LLM usage, token, latency, and cost accounting.

```text
id TEXT PK
matter_id TEXT
run_id TEXT
model_tier TEXT
model_id TEXT
usage_label TEXT
input_tokens INTEGER
cache_read_tokens INTEGER
output_tokens INTEGER
total_prompt_tokens INTEGER
estimated_cost_usd REAL
latency_ms INTEGER
success INTEGER
error_kind TEXT
created_at TEXT
prompt_hash TEXT
response_hash TEXT
tool_use_prompt_tokens INTEGER
thinking_tokens INTEGER
```

### `matter`

Purpose: top-level workspace metadata.

```text
id TEXT PK
name TEXT
repository_root TEXT
forum TEXT
posture TEXT
governing_law TEXT
maturity TEXT
trust_revision INTEGER
created_at TEXT
updated_at TEXT
```

### `migration_backfill_job`

Purpose: durable background migration/backfill work tracking.

```text
id TEXT PK
matter_id TEXT
job_name TEXT
status TEXT
priority INTEGER
total_items INTEGER
completed_items INTEGER
estimated_tokens INTEGER
estimated_cost_usd REAL
error TEXT
created_at TEXT
started_at TEXT
completed_at TEXT
```

### `pending_propagation`

Purpose: deferred belief propagation work.

```text
id TEXT PK
matter_id TEXT
assertion_id TEXT
cause TEXT
orig_run_id TEXT
queue TEXT
enqueued_at TEXT
```

### `proof_state`

Purpose: stored proof coverage snapshot for an issue.

```text
id TEXT PK
matter_id TEXT
issue_id TEXT
sufficiency REAL
supporting_count INTEGER
attacking_count INTEGER
total_predicate_count INTEGER
satisfied_predicate_count INTEGER
proof_status TEXT
support_score REAL
attack_score REAL
coverage_version TEXT
notes TEXT
computed_at TEXT
trust_weighted_support REAL
trust_weighted_attack REAL
advocacy_only INTEGER
```

### `provenance_event`

Purpose: append-only derivation metadata.

```text
id TEXT PK
matter_id TEXT
target_kind TEXT
target_id TEXT
event_kind TEXT
writer_name TEXT
run_id TEXT
model_id TEXT
model_tier TEXT
prompt_version TEXT
extractor_version TEXT
llm_call_id TEXT
prompt_hash TEXT
response_hash TEXT
source_document_ref TEXT
source_document_inventory_id TEXT
source_span_id TEXT
source_span_status TEXT
note TEXT
created_at TEXT
```

### `quant_fact`

Purpose: structured quantitative fact.

```text
id TEXT PK
matter_id TEXT
quant_kind TEXT
amount_value REAL
date_value TEXT
date_end_value TEXT
rate_value REAL
currency TEXT
unit TEXT
raw_text TEXT
subject_type TEXT
subject_id TEXT
span_id TEXT
assertion_id TEXT
date_precision TEXT
quant_dedup_key TEXT
created_at TEXT
```

### `reasoning_cache`

Purpose: scoped cache for routing and reasoning plans.

```text
id TEXT PK
matter_id TEXT
stage TEXT
cache_key TEXT
plan_json TEXT
created_at TEXT
last_hit_at TEXT
```

### `run_session`

Purpose: durable run lifecycle, objective, steering flags, and reuse/cost metrics.

```text
id TEXT PK
matter_id TEXT
query TEXT
objective TEXT
active_branch_issue_id TEXT
status TEXT
stop_requested INTEGER
redirect_requested INTEGER
next_action TEXT
operation_type TEXT
trigger TEXT
research_mode TEXT
started_at TEXT
completed_at TEXT
resumed_from TEXT
assertions_at_start INTEGER
reuse_rate REAL
llm_calls_avoided INTEGER
llm_calls_required INTEGER
llm_input_tokens INTEGER
llm_cache_read_tokens INTEGER
llm_output_tokens INTEGER
llm_request_count INTEGER
llm_estimated_cost_usd REAL
llm_tool_use_prompt_tokens INTEGER
llm_thinking_tokens INTEGER
llm_total_processed_tokens INTEGER
```

### `schema_migration`

Purpose: migration audit.

```text
version INTEGER PK
name TEXT
checksum TEXT
applied_at TEXT
app_schema_version INTEGER
app_build TEXT
duration_ms INTEGER
```

### `schema_version`

Purpose: current schema version marker.

```text
version INTEGER
applied_at TEXT
```

### `span`

Purpose: stable source location and extracted text.

```text
id TEXT PK
document_id TEXT
span_type TEXT
page_start INTEGER
page_end INTEGER
line_start INTEGER
line_end INTEGER
char_start INTEGER
char_end INTEGER
section_ref TEXT
clause_ref TEXT
parent_span_id TEXT
ordinal_in_doc INTEGER
text_hash TEXT
span_text TEXT
parser_version TEXT
created_at TEXT
```

### `verification_event`

Purpose: append-only review lifecycle event.

```text
id TEXT PK
matter_id TEXT
verification_id TEXT
target_kind TEXT
target_id TEXT
old_status TEXT
new_status TEXT
reviewed_by_kind TEXT
reviewed_by_id TEXT
review_scope TEXT
rejection_reason TEXT
run_id TEXT
cause TEXT
note TEXT
old_version INTEGER
new_version INTEGER
created_at TEXT
```

### `verification_state`

Purpose: current review state for a target object.

```text
id TEXT PK
matter_id TEXT
target_kind TEXT
target_id TEXT
status TEXT
ai_confidence REAL
reviewed_by_kind TEXT
reviewed_by_id TEXT
reviewed_at TEXT
review_scope TEXT
review_scope_json TEXT
review_note TEXT
rejection_reason TEXT
stale_reason TEXT
version INTEGER
created_at TEXT
updated_at TEXT
```

## 6. Core Enums And Vocabularies

These are hardcoded today. A general ontology mapping system should move most of these into a domain profile.

### SpeechAct

```text
alleged
argued
denied
admitted
ordered
performed
paid
requested
threatened
promised
estimated
calculated
observed
testified
stipulated
amended
waived
terminated
inferred
operative
extracted
```

Meaning: what kind of source act produced the assertion. This influences initial belief state.

### SourceRole

```text
advocacy
operative
procedural
authoritative
informal
draft
post_hoc
unknown
```

Current trust weights:

```text
operative: 1.0
authoritative: 1.0
procedural: 0.7
informal: 0.5
unknown: 0.5
draft: 0.4
advocacy: 0.3
post_hoc: 0.3
```

### BeliefState

```text
alleged
argued
admitted
operative
performed
not_performed
disputed
superseded
withdrawn
inferred
resolved
unknown
```

Meaning: current truth-maintenance state of a canonical assertion.

### ModelLayer

```text
record
reality
proof
legal
decision_context
```

Meaning: what layer the assertion belongs to. For generalization, `legal` should become a domain-specific normative or rule layer.

### AssertionKind

```text
factual
normative
quantitative
temporal
relational
```

### AssertionLinkType

```text
supports
attacks
depends_on
supersedes
contradicts
corroborates
```

### RevisionCause

```text
new_evidence
user_correction
conflict_detection
supersession
admission
trust_override
```

### GapType

```text
missing_document
missing_metadata
missing_issue_predicate
missing_authority
missing_user_context
missing_quantitative_input
unresolved_contradiction
expected_absent_attachment
expected_absent_notice
```

### IssueType

```text
claim
defense
contract_question
condition_precedent
waiver
damages
diligence_red_flag
compliance_failure
procedural_barrier
evidentiary_bottleneck
```

### OriginKind

```text
extracted
inferred
user_supplied
imported
```

### RunStatus

```text
running
completed
interrupted
failed
```

### LedgerEventType

```text
run_started
objective_set
branch_selected
assertion_added
assertion_revised
conflict_detected
gap_identified
issue_updated
user_interrupted
user_redirected
user_correction
synthesis_started
run_completed
run_failed
system_warning
progress_note
route_decision
```

### VerificationStatus

```text
candidate
verified
rejected
stale
```

### VerificationTargetKind

```text
assertion
assertion_occurrence
issue_predicate
evidence_edge
quant_fact
authority
document_card
privilege_classification
gap
dispute
dispute_position
timeline_event
deadline
authority_treatment
actor_relationship
defined_term
causation_edge
theory
artifact
artifact_manifest_item
```

Not all target kinds have full runtime implementations today. Some are reserved vocabulary for planned objects.

### ReviewScope

```text
extraction_correct
record_truth
inference
legal_conclusion
truth_override
internal_privileged
clean_output
privilege_classification
dispute_resolution
artifact_policy
```

### ReviewedByKind

```text
user
attorney
system
import
```

### EvidenceRelationType

```text
supports
establishes
attacks
negates
```

### EvidenceOriginKind

```text
ai_extracted
attorney_annotated
system_inferred
imported
legacy_backfill
```

## 7. Assertion Identity And Fact Reuse

The most important implemented reuse mechanism is canonical assertion identity.

An extraction produces an `AssertionCandidate`. The candidate is canonicalized into:

- `proposition_key`: legacy normalized text hash.
- `claim_key`: structured identity hash.
- `identity_version`: current identity algorithm version.
- structured subject, predicate, object, temporal scope, polarity, and speaker scope fields.

The store then upserts into `assertion` and records one `assertion_occurrence`.

### Identity tiers

The identity resolver uses three levels.

#### Tier A: structured subject-predicate-object identity

Used when subject, predicate, and object are available.

Identity input:

```text
model_layer
assertion_kind
polarity
canonical_subject_key
predicate_key
canonical_object_key
temporal_identity_key
```

Speaker scope is intentionally empty in this tier.

Canonicalization confidence is at least `0.95` unless a higher extraction confidence is supplied.

#### Tier B: partial structured identity

Used when the predicate is available and at least one side of the subject/object structure is available.

Identity input includes:

```text
model_layer
assertion_kind
polarity
known subject or text fallback
predicate_key
known object or text fallback
temporal_identity_key
speaker_scope_key or speaker:unknown
```

Canonicalization confidence is at least `0.70`.

#### Tier C: text signature fallback

Used when structured identity is not enough.

Identity is based on:

```text
normalized text signature
unresolved predicate placeholder
speaker scope
```

Canonicalization confidence is at least `0.40`.

### Subject and object keys

Structured references use:

```text
<ref_type>:<ref_id>
```

Free text is converted into a normalized phrase hash:

```text
np:<hash>
```

Objects are parsed from JSON when possible. If an object is a structured reference with `ref_type` and `ref_id`, the same structured reference key is used. Otherwise the stable JSON representation is hashed.

### Temporal identity

Temporal identity is:

```text
<start>|<end>
```

or:

```text
atemporal
```

### Speaker scope

Speaker scope can come from:

- explicit `speaker_scope_key`
- `speaker_actor_id`
- source side
- no speaker scope

Speaker scope matters most for partial or text-fallback identity, where the same sentence can have different meaning depending on who said it.

### Assertion upsert behavior

`AssertionStore.upsert_occurrence()` implements this invariant:

```text
one canonical assertion
many assertion occurrences
```

If a new occurrence maps to an existing assertion, the existing assertion can be enriched:

- better structured subject fields
- better predicate key
- better object JSON
- better temporal fields
- stronger confidence
- stronger belief state when appropriate

These changes are audited through `assertion_revision`.

### Initial belief state from speech act

New assertions receive an initial belief state from the source act:

```text
operative -> operative, confidence 0.8
admitted/stipulated -> admitted, confidence 0.8
performed/paid -> performed, confidence 0.8
inferred -> inferred, confidence 0.6
alleged/argued -> alleged or argued, confidence 0.3
waived/terminated/amended -> operative, confidence 0.7
default -> unknown, confidence 0.5
```

Existing assertions are not blindly overwritten. Terminal states such as `superseded` and `withdrawn` are protected from casual upgrades.

## 8. Assertion Occurrences

`assertion_occurrence` is the source-facing side of the claim system.

It stores:

- source document identity
- source span
- raw source text
- speaker
- source role
- source side
- speech act
- origin kind
- structured subject/predicate/object fields
- temporal scope
- polarity
- candidate claim key
- resolution strategy
- extraction confidence

This separation is why the same fact can be reused across many documents or many extractions. The system does not need to duplicate the canonical claim every time the claim appears.

### Occurrence deduplication

The occurrence writer is conservative:

- If a `span_id` exists, span-aware deduplication is allowed.
- Without a span, broad collision is avoided.
- The system prefers extra occurrences over incorrectly merging different source mentions.

## 9. Belief Revision And Conflict Propagation

`BeliefRevisionEngine` is the truth-maintenance engine over `assertion_link`.

It is responsible for:

- applying forced user/system state changes
- recomputing dependent assertions
- propagating conflict effects
- recording field-level revisions
- recording belief revision events
- deferring unfinished propagation work

### Graph shape

Assertions are nodes. `assertion_link` rows are directed edges.

Important edge types:

- `supports`: source assertion supports target assertion.
- `attacks`: source assertion attacks target assertion.
- `contradicts`: source assertion contradicts target assertion.
- `supersedes`: source assertion supersedes target assertion.
- `depends_on`: source assertion depends on target assertion.
- `corroborates`: source assertion corroborates target assertion.

### Work budget

The default propagation budget is:

```text
MAX_WORK = 500
```

For large seed sets, the effective budget can scale up to:

```text
max(500, seed_count * 3)
```

with a ceiling of:

```text
2000
```

Unvisited nodes can be stored in `pending_propagation`.

### Optimistic concurrency

The engine retries optimistic write conflicts up to three times. If it cannot safely update a node, it can abandon that node for this pass and leave work for a later flush.

### Inactive states

These states are treated as inert for support/attack propagation:

```text
withdrawn
superseded
unknown
```

These states undermine support:

```text
disputed
withdrawn
superseded
unknown
```

These states are promoting support:

```text
operative
admitted
resolved
performed
```

### Supersession behavior

If an active assertion supersedes another assertion, the older assertion becomes `superseded`.

If superseding links later become inert, the target can recover from `superseded` unless a user lock explicitly set that state.

### Dispute behavior

If active attacks exist and strong support does not overcome them, the target becomes `disputed`.

If all real attack links later become inert, the assertion can recover from `disputed` unless a user lock explicitly set that state.

### Support behavior

If support exists but all support collapses, the target can become `unknown` with low confidence.

If promoting support exists and there are no attacks, the target can become `inferred` and confidence can rise up to `0.9`.

If strong non-promoting support exists and there are no attacks, the engine may keep the current belief state but boost confidence up to `0.8`.

### Attack behavior

If attacks are active and support is weak:

```text
new state: disputed
confidence: max(0.1, 0.5 - 0.1 * attack_weight)
```

If attacks and strong support both exist:

```text
new state: disputed
confidence: approximately 0.3 to 0.5 depending on support fraction
```

### Audit behavior

Belief changes write:

- `assertion_revision` rows for changed fields.
- `belief_revision_event` rows for state and confidence changes.

The engine also detects oscillation and avoids unbounded loops.

## 10. Contradiction Detection

There are two contradiction mechanisms.

### Explicit graph contradiction

`AssertionStore.find_contradictions()` returns pairs linked by `attacks` or `contradicts` where both endpoints are active.

Inactive endpoints are excluded:

```text
superseded
withdrawn
resolved
```

### Heuristic contradiction mining

`detect_heuristic_contradictions()` checks active assertions linked to open issues.

The heuristic compares:

- significant token overlap
- conservative negation asymmetry
- per-issue caps to avoid runaway comparisons

Current threshold:

```text
token Jaccard >= 0.4
at least 2 shared significant tokens
max 50 assertions per issue
```

When detected, the system creates a `contradicts` link from the negated assertion to the positive assertion.

### Conflict marking

`mine_and_mark_contradictions()` can mark an attacked assertion as `disputed` when:

- the attacker is `operative` or `admitted`
- the attacked assertion is `operative`, `alleged`, `argued`, or `inferred`

The system also records an `unresolved_contradiction` gap for unresolved conflicts.

## 11. Issues, Predicates, Evidence, And Proof

The proof system has two layers:

- `issue` and `issue_predicate`: target-side objective structure.
- `evidence_edge`: source-to-target proof graph.

### Issues

An `issue` is currently legal-labeled, but structurally it is an objective node.

It can represent:

- a legal issue
- a factual question
- a reasoning target
- a concept to map
- a topic in an ontology
- a task objective
- a compliance requirement
- a hypothesis

Issues are hierarchical through `parent_issue_id`.

### Issue predicates

An `issue_predicate` is a testable element beneath an issue.

In legal mode, examples are elements of a claim. In ontology mode, these could be:

- required properties
- slots
- constraints
- mapping criteria
- class membership tests
- relation requirements

### Built-in templates

Current built-in templates are legal:

```text
contract_breach v1
  formation
  performance_or_excuse
  breach
  causation
  damages

negligence v1
  duty
  breach
  causation
  damages
```

`IssueStore.apply_template()` materializes the template elements as `issue_predicate` rows and seeds candidate verification rows.

For a general-purpose ontology system, these templates should become domain profile data.

### Assertion-to-issue links

`IssueStore.link_assertion()` writes:

1. A legacy `assertion_issue_link`.
2. A companion `evidence_edge` when the relation is proof-relevant.

Proof-relevant relation types:

```text
supports
establishes
attacks
negates
```

### Assertion-to-predicate links

`link_assertion_to_predicate()` writes an `evidence_edge` targeting `issue_predicate`.

It can also mirror the relationship up to the parent issue by calling `link_assertion()`.

### Evidence edge behavior

`EvidenceStore` is the canonical proof substrate.

Supported source kinds are open-ended strings. Production use is mostly:

```text
assertion -> issue
assertion -> issue_predicate
```

Other intended source or target kinds include:

```text
authority
work_product
quant_fact
document_card
artifact
```

On upsert, the store:

- preserves reviewed state
- refreshes weights for unreviewed system-inferred or AI-extracted candidate edges
- seeds or touches verification state
- can record provenance

### Proof state snapshots

`ProofStateStore` computes and stores one proof snapshot per issue.

Thresholds:

```text
PARTIAL_THRESHOLD = 0.25
SUFFICIENT_THRESHOLD = 0.75
```

Proof statuses:

```text
insufficient
partial
sufficient
contested
```

General rules:

- zero support or support below `0.25` is insufficient
- support from `0.25` to `0.75` is partial
- support above `0.75` is sufficient
- if attacks are at least as numerous as supports and support exists, status is contested

The proof snapshot tracks:

- raw supporting count
- raw attacking count
- total predicate count
- satisfied predicate count
- support score
- attack score
- trust-weighted support
- trust-weighted attack
- advocacy-only flag
- notes

### Advocacy-only detection

`advocacy_only` is true when all supporting assertions have effective trust at or below:

```text
0.35
```

In legal mode this means the issue is supported only by advocacy-like sources. In a general ontology system, this becomes "supported only by low-trust sources."

### Clean mode proof filtering

Clean mode excludes assertions sourced from privileged documents.

Rejected and stale assertions or edges are excluded from proof state.

### Issue coverage report

`MatterModel.get_issue_coverage_report()` computes a richer coverage report over open issues.

It prefers `evidence_edge` when an issue has active evidence edges. Otherwise it falls back to `assertion_issue_link`.

Belief state weights:

```text
operative/admitted/resolved -> 1.0
alleged/argued/inferred -> 0.5
other active -> 0.3
disputed/withdrawn/superseded -> excluded
```

If predicates exist:

```text
coverage_fraction = min(weighted_support, predicate_count) / predicate_count
```

If no predicates exist:

```text
coverage_fraction = weighted_support / (weighted_support + 1.0)
```

The report includes:

- supporting count
- attacking count
- verified supporting count
- candidate supporting count
- verified coverage fraction
- proof status
- proof gaps
- contested predicates
- blocked predicates
- hierarchy metadata
- subtree coverage

For edge-backed rows, verified support requires both assertion and edge verification lanes to be verified.

## 12. Verification, Trust, Policy, And Provenance

The system separates four related concerns:

- Verification: whether a specific target has been reviewed.
- Trust: whether the target is usable for a purpose.
- Content policy: whether content can be shown or sent to the model.
- Provenance: how the target was produced.

### VerificationStateStore

Current statuses:

```text
candidate
verified
rejected
stale
```

Important behavior:

- `candidate` is idempotent.
- candidate creation does not downgrade verified or rejected rows.
- `verify` requires `reviewed_by_kind` of `user` or `attorney`.
- `reject` requires a human reviewer and a rejection reason.
- `mark_stale` can be done by the system for verified or candidate rows.
- rejected rows are not downgraded by stale marking.
- `touch_ai_target` creates candidate rows or refreshes stale rows to candidate.
- when target kind is `evidence_edge`, verification status mirrors into `evidence_edge.verification_status`.
- every transition writes `verification_event`.

### Review queue

The review queue prioritizes candidate targets roughly in this order:

1. proof-gap assertions and edges
2. contradicted assertions
3. issue-linked assertions and edges
4. issue predicates
5. quantitative facts
6. authorities
7. other candidate targets

### Trust policy

`TrustPolicy` maps verification and other status signals into trust buckets.

Trust buckets:

```text
verified
candidate
stale
excluded
```

Exclusion precedence:

1. clean-mode privilege exclusion
2. inactive belief state
3. rejected verification status
4. stale verification status
5. verified if all relevant lanes are verified
6. candidate otherwise

### Trust purposes

The policy changes by purpose:

```text
HYDRATION -> all buckets can be visible to internal state construction
CACHED_SEARCH -> verified and candidate
STRUCTURED_READ -> verified and candidate
PROOF_CANDIDATE -> verified and candidate
PROOF_VERIFIED -> verified only
SYNTHESIS_DEFINITIVE -> verified only
```

### Content policy

Content policy decides whether content can be used for a concrete purpose.

Purposes:

```text
profile
deep_read
search_snippets_to_llm
hydration
synthesis_context
timeline_view
matrix_view
chat_response
export
```

Actions:

```text
allow
block
withhold
```

Clean mode behavior:

- privileged content is withheld for timeline, matrix, export, hydration, search snippets, and synthesis context
- privileged content is blocked for other external purposes
- internal audience can be allowed

After privilege handling, content policy applies trust policy:

- rejected, inactive, or stale content is blocked or withheld depending on purpose
- verified and candidate content can be allowed with reason codes

`ContentPolicyGuard` wraps this and appends `content_policy_audit` rows.

### Privilege gate

`PrivilegeGate` checks:

- whether a document is privileged
- whether an assertion has privileged sources

In a general system, privilege becomes sensitivity, access control, or visibility classification.

### Provenance store

`ProvenanceStore` appends `provenance_event` rows.

An AI-derived object can carry:

- writer name
- run id
- model id
- model tier
- prompt version
- extractor version
- LLM call id
- prompt hash
- response hash
- source document reference
- source document inventory id
- source span id
- source span status
- note

This is how the system can explain where a fact, edge, or profile came from.

## 13. Matter Runtime Adapter

`MatterRuntimeAdapter` bridges the recursive investigation engine to the durable matter model.

### Context hydration

At orientation, `get_context()` builds a `QueryMatterContext` from durable state.

`MatterModel.build_query_context()` includes:

- matter id and name
- open gaps with materiality at least `0.3`
- open issues
- current counts
- known actors
- known documents from assertion occurrences
- weakest issue id based on materiality, salience, and coverage
- answered clarifications
- document annotations
- top predicate keys
- active assumptions
- document card count

This is the main reason the model reuses facts. It starts with a compact map of known state.

### Recording facts

`record_fact()` performs the following work:

1. Infer `SourceRole` from filename if the source role is unknown.
2. Adjust speech act:
   - `EXTRACTED` plus advocacy source becomes `ALLEGED`.
   - `EXTRACTED` plus operative or authoritative source becomes `OPERATIVE`.
3. Apply trust override:
   - low trust can keep or push toward alleged
   - high trust can promote extracted or alleged facts to operative
4. Choose model layer:
   - normative facts become `LEGAL`
   - inferred facts become `REALITY`
   - otherwise facts default to `RECORD`
5. Resolve speaker actor from document card sender when possible.
6. Build `AssertionCandidate`.
7. Attach provenance context.
8. Upsert canonical assertion and occurrence.
9. Append ledger event if this is a new assertion.
10. If an issue id and proof relation are present, link the assertion to the issue with an evidence edge.

### Recording quantitative facts

`record_quant()` writes `quant_fact` rows and can attach provenance.

The quant store supports reconciliation patterns such as:

- subject-level facts
- payment chains
- invoice chains
- conflicts
- thresholds

### Flushing revisions

`flush_revisions()` drains pending belief propagation and recomputes affected proof states.

## 14. Cascade Governor And Routing

`CascadeGovernor` chooses the cheapest sufficient route for a query.

### ExecutionContract

A route has an execution contract:

```text
family
min_iter
max_iter
citation_floor
answer_confidence_floor
escalation_allowed
lead_ev_floor
```

### AnswerabilitySnapshot

The route classifier receives:

```text
matter_id
assertion_count
verified_assertion_count
open_issue_count
open_gap_count
actor_count
has_any_facts
has_any_verified
trust_revision
policy_audience
recent turn count
last turn summary
```

### Cold start rule

If there are no facts, the route is hard-routed to full investigation.

### Route cache

The classifier result can be cached in `reasoning_cache`.

The cache key includes:

- normalized query
- snapshot fingerprint
- classifier schema version
- trust revision
- schema version

Current classifier schema version:

```text
mvi7.0
```

The cache is invalidated by trust or schema changes so stale routing plans do not survive meaningful state changes.

## 15. RLM Engine

`RLMEngine` is the recursive lead-mining engine.

Important configuration values:

```text
max_depth: 5
max_leads_per_level: 5
max_documents_per_search: 10
min_lead_priority: 0.3
excerpt_chars: 8000
parallel_reads: 5
adaptive_depth: true
min_depth: 2
depth_citation_threshold: 15
max_iterations: 20
enable_matter_model: true
```

Packet budget:

```text
coverage_tokens: 256
gap_tokens: 256
orientation_tokens: 800
per_optional_section_tokens: 384
synthesis_total_tokens: 3000
```

The engine runs stages broadly like:

```text
orient
investigate loop
verify citations
synthesize
```

The exact prompts are separate from the schema. The implemented architectural point is that the engine records extracted structure into the durable matter graph through the runtime adapter.

## 16. Ledger, Runs, Cost, And Reuse Accounting

`ReasoningLedgerStore` manages run lifecycle.

It supports:

- starting runs
- completing runs
- failing runs
- interrupting runs
- appending ordered ledger events
- stop flags
- redirect flags
- resumable `next_action` checkpoints

Run completion stores:

- assertions at start
- reuse rate
- LLM calls avoided
- LLM calls required
- input tokens
- cache-read tokens
- output tokens
- request count
- estimated cost
- tool-use prompt tokens
- thinking tokens
- total processed tokens

This is the accounting layer that makes fact reuse measurable.

## 17. Documents, Spans, Actors, Gaps, Assumptions, And Quantities

### Documents

The document subsystem has:

- `document_inventory`: file/path/hash/status/family/version/salience inventory.
- `document_card`: profiled metadata.
- `span`: source ranges and text hashes.
- `document_actor_role`: actors mentioned in or related to a document.
- `document_relation`: links between documents.

Version detection can create family and version-chain metadata and gaps for missing bases.

### Actors

The actor subsystem has:

- canonical actors
- aliases
- affiliations
- document roles

Resolution by name uses normalized names and aliases. The current implementation is good enough for practical entity reuse, but not a full ontology-grade entity resolution engine.

### Gaps

Gaps capture missingness and blockers.

Each gap has:

- type
- description
- expected artifact
- materiality score
- blocker score
- status
- optional resolution note

Gaps can be linked to affected objects through `gap_link`.

`generate_clarifications_from_gaps()` creates targeted questions with:

- question text
- why it matters
- expected impact

It avoids duplicate pending questions.

### Assumptions

Assumptions are explicit provisional statements.

They can be:

- linked to target objects
- listed as active
- confirmed
- invalidated
- counted
- checked for unresolved blockers

Scenario routing can apply temporary assumptions for a turn, but fully durable scenario branching is not implemented as a separate versioned world model.

### Quantitative facts

`quant_fact` stores structured numeric, date, rate, currency, and unit facts.

It includes:

- quant kind
- amount
- date or date range
- rate
- currency
- unit
- raw text
- subject type and id
- span id
- assertion id
- date precision
- dedup key

The quant store supports domain-specific reconciliation patterns that are currently legal/commercial in flavor.

## 18. How The System Ensures Reuse

The implemented reuse strategy has several layers.

### Claim-level reuse

`claim_key` prevents repeated extraction of the same structured claim from becoming duplicate canonical facts.

### Occurrence-level reuse

`assertion_occurrence` keeps all source mentions attached to the same canonical assertion.

### Context-level reuse

`build_query_context()` gives the next run a compact inventory of:

- known facts
- known issues
- known gaps
- known actors
- known documents
- known predicates
- active assumptions

### Route-level reuse

`CascadeGovernor` can answer from existing durable state or choose a lighter route when the snapshot says the matter is answerable.

### Plan-level reuse

`reasoning_cache` stores route and plan decisions scoped by schema and trust revision.

### Proof-level reuse

`proof_state` stores proof coverage snapshots, so issue weakness and gaps can be revisited without recomputing all structure from scratch.

### Review-level reuse

`verification_state` prevents verified or rejected objects from being casually overwritten by later AI runs.

## 19. How The System Propagates Conflicts

Conflict propagation happens through three implemented channels.

### Assertion links

Contradiction, attack, supersession, dependency, and support links let the engine recompute downstream assertion belief states.

### Issue proof state

When assertions or evidence edges attack an issue, proof status can become contested.

### Gaps

Unresolved contradictions create `gap` rows. This turns conflict into visible work, not just hidden graph state.

The important design property is that conflict is not only a response-time narrative. It becomes durable state.

## 20. General-Purpose Ontology Mapping Direction

The current implementation suggests a strong general architecture:

```text
stable reasoning kernel
  plus domain profile
  plus domain adapters
  plus profile-defined templates
  plus stable review/provenance/policy
```

### Avoid fully dynamic database schemas

Letting every user define arbitrary primitives and then generating new tables sounds flexible, but it creates infrastructure problems:

- migrations become dynamic and hard to reason about
- query planning becomes unpredictable
- proof algorithms need custom SQL per domain
- review queues become domain-specific code
- caches become harder to invalidate safely
- portability between projects gets worse
- stable APIs become difficult

The better architecture is a stable meta-model with configurable vocabularies.

### Recommended universal primitives

These are the portable concepts worth keeping as the base layer:

```text
Workspace
Corpus
Artifact
Span
Entity
Claim
ClaimOccurrence
Relation
ObjectiveNode
Criterion
SupportEdge
Gap
ReviewState
ProvenanceEvent
PolicyDecision
Run
Metric
```

Mapping from current names:

```text
matter -> Workspace
document_inventory -> Artifact
document_card -> ArtifactProfile
span -> Span
actor -> Entity
assertion -> Claim
assertion_occurrence -> ClaimOccurrence
assertion_link -> ClaimRelation
issue -> ObjectiveNode
issue_predicate -> Criterion
evidence_edge -> SupportEdge
gap -> Gap
verification_state -> ReviewState
verification_event -> ReviewEvent
provenance_event -> ProvenanceEvent
run_session -> Run
ledger_event -> RunEvent
llm_call -> ModelCall
reasoning_cache -> ReasoningCache
content_policy_audit -> PolicyDecisionAudit
```

### Domain profile instead of hardcoded legal enums

A domain profile should define:

- entity types
- claim kinds
- model layers
- relation types
- source roles
- trust weights
- evidence relation types
- objective node types
- criterion templates
- review scopes
- policy classes
- artifact types
- renderable deliverables

Example profile shape:

```json
{
  "profile_id": "general_ontology_mapping",
  "entity_types": ["concept", "term", "source", "standard", "dataset", "person", "organization"],
  "claim_kinds": ["definition", "classification", "mapping", "constraint", "measurement", "temporal", "causal"],
  "model_layers": ["record", "semantic", "inference", "normative", "decision_context"],
  "relations": ["supports", "attacks", "contradicts", "supersedes", "depends_on", "maps_to", "broader_than", "narrower_than"],
  "source_roles": {
    "canonical_source": 1.0,
    "primary_record": 0.9,
    "curated_reference": 0.8,
    "secondary_analysis": 0.6,
    "unreviewed_import": 0.4,
    "user_note": 0.5
  }
}
```

### Ontology templates

The current `issue_predicate` template mechanism can generalize directly.

Example:

```text
template: map_external_term_to_internal_concept
criteria:
  source term exists
  source term definition captured
  candidate internal concept exists
  semantic overlap established
  conflicting mappings checked
  source provenance captured
  human review completed
```

Each criterion becomes an `issue_predicate`. Claims and evidence edges then support or attack those criteria.

### Claim identity for ontology mapping

The existing claim identity resolver should generalize by using:

```text
subject = source concept, term, entity, or artifact
predicate = ontology relation or property key
object = target concept, value, constraint, or entity
temporal scope = version, validity period, source date, or schema version
speaker scope = source authority or contributor when needed
model layer = record, semantic, inference, normative, or decision context
```

Examples:

```text
source_term:abc maps_to internal_concept:def
source_term:abc has_definition "..."
concept:def narrower_than concept:xyz
dataset:claims uses_code_system icd10
standard:foo requires_property bar
```

### Conflict propagation for ontology mapping

The existing conflict engine can handle ontology conflicts:

- two mappings contradict each other
- a newer ontology version supersedes an older mapping
- a canonical source attacks an imported mapping
- a reviewed definition supports a classification
- a concept merge invalidates downstream edges

The current mechanism is already close to a truth-maintenance system for ontology graphs.

## 21. Suggested Base-Layer Schema For A General Product

The current schema can be generalized without losing implementation leverage.

Recommended base tables:

```text
workspace
artifact
artifact_profile
span
entity
entity_alias
entity_relation
claim
claim_occurrence
claim_relation
objective_node
criterion
support_edge
gap
gap_link
assumption
assumption_link
review_state
review_event
provenance_event
policy_decision_audit
run
run_event
model_call
reasoning_cache
domain_profile
domain_vocabulary
domain_template
domain_template_element
```

Keep these as stable infrastructure. Let users define domain vocabulary rows, not arbitrary physical tables.

### Why this works

The reasoning algorithms only need stable graph primitives:

- claims
- claim relations
- support edges
- objective nodes
- criteria
- review state
- provenance
- policy
- gaps

Everything else can be vocabulary.

## 22. Current Limitations And Non-Implemented Pieces

This section is intentionally explicit so the portable document does not overstate the implementation.

### Legal naming is still embedded

The live schema and Python APIs use legal-flavored terms. General-purpose ontology support is not implemented as a first-class profile system today.

### Dynamic domain profiles are not implemented

There is no implemented `domain_profile` table yet. Current vocabularies are Python enums and template registry code.

### `authority` is legal-specific

Authority handling exists as storage and issue-linking, but a general reference-source abstraction is not implemented.

### `evidence_link` is legacy

The canonical current proof substrate is `evidence_edge`. `evidence_link` remains for compatibility and should not be treated as the future-facing model.

### Some verification target kinds are reserved

The enum includes target kinds such as `deadline`, `dispute`, `defined_term`, `causation_edge`, and `artifact_manifest_item`, but not all have complete runtime stores.

### OCR is not the core implemented layer

The reasoning schema assumes source text and spans can exist. End-to-end OCR is not the important implemented primitive described here.

### Scenario branching is shallow

Scenario routing can apply temporary assumptions, but the system does not yet implement persistent multi-world scenario branches with independent truth graphs.

### Conflict detection is partly heuristic

Explicit links are durable and meaningful. Heuristic contradiction mining is conservative and issue-local.

### Entity resolution is practical, not exhaustive

Actor aliasing and normalized-name resolution exist. A full ontology-grade entity resolution subsystem would need more work.

## 23. Design Takeaways

The strongest implemented idea is not "legal AI." It is:

```text
durable claim graph
plus occurrence-backed provenance
plus issue/criterion proof targets
plus review state
plus conflict propagation
plus route and cost governance
```

To make this general purpose:

1. Keep the stable graph kernel.
2. Rename legal nouns to neutral nouns.
3. Move enum values and templates into domain profiles.
4. Treat ontology schemas as data that configure the graph, not as generated database schemas.
5. Keep provenance, review, trust, and policy as first-class infrastructure.
6. Let domain adapters translate raw artifacts into claims, occurrences, entities, criteria, and support edges.

The current system is already close to a general reasoning substrate. The main architectural step is to separate the reasoning kernel from the legal vocabulary.

## 24. Minimal Portable Vocabulary

If this were explained to a new model or another implementation team, use these terms:

```text
Workspace: durable context for a reasoning project.
Artifact: source object being reasoned over.
Span: stable range inside an artifact.
Entity: canonical participant, concept, object, source, or organization.
Claim: canonical proposition.
ClaimOccurrence: one source mention of a claim.
ClaimRelation: support, attack, contradiction, dependency, supersession, or corroboration between claims.
ObjectiveNode: issue, question, concept, task, or mapping target.
Criterion: testable requirement under an objective node.
SupportEdge: source object supporting or attacking a target object.
Gap: missingness or unresolved conflict.
ReviewState: current human/system review status.
ProvenanceEvent: how an object was produced.
PolicyDecision: whether content can be used for a purpose.
Run: one reasoning session.
RunEvent: ordered audit event inside a run.
ModelCall: cost and token accounting for an LLM call.
```

## 25. Minimal Kernel Algorithms

A general-purpose implementation should preserve these algorithms.

### Claim canonicalization

```text
input extraction
  -> normalize subject/predicate/object/time/speaker
  -> compute claim_key
  -> upsert canonical claim
  -> insert claim occurrence
  -> preserve source provenance
```

### Belief propagation

```text
changed claim
  -> traverse claim relations
  -> recompute affected belief states
  -> write revision audit
  -> defer unfinished work if budget exceeded
```

### Proof coverage

```text
objective node
  -> collect criteria
  -> collect support and attack edges
  -> filter by review/trust/policy
  -> compute coverage and contested status
  -> store proof snapshot
```

### Gap generation

```text
weak coverage, missing criteria, missing source, or contradiction
  -> create gap
  -> link gap to affected object
  -> optionally generate clarification question
```

### Review lifecycle

```text
AI-created object
  -> candidate
  -> human verifies, rejects, or later system marks stale
  -> policy uses review state to decide allowed uses
```

### Route governance

```text
query
  -> build answerability snapshot
  -> use cache if valid
  -> choose cheapest sufficient route
  -> escalate only if needed
  -> record route decision
```

## 26. Practical Migration Path From Current Legal System

Recommended sequence:

1. Introduce neutral aliases in documentation and API boundaries:
   - matter/workspace
   - assertion/claim
   - issue/objective node
   - issue predicate/criterion
   - evidence edge/support edge
2. Add a `domain_profile` concept without changing the core graph.
3. Move issue templates from Python-only legal templates into profile data.
4. Add profile-defined source roles and trust weights.
5. Add profile-defined claim kinds, model layers, and relation types.
6. Keep current legal profile as one installed domain profile.
7. Build an ontology mapping profile as a second profile.
8. Refactor UI/API labels after the data model can support both profiles.

This path avoids a risky rewrite. It preserves the implemented reuse, provenance, conflict, review, and proof machinery while making the vocabulary configurable.
