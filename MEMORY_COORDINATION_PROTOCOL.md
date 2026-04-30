# Memory Coordination Protocol

Status: active design loop, not yet implementation complete.
Last updated: 2026-04-30.

## Purpose

Irys should become a durable long-context reasoning substrate, not a chatbot
with a larger prompt. The legal matter model is the first installed domain, but
the target architecture must also support finance, code, neuroscience,
medicine, research review, and other ontology-mapping workloads.

The memory layer must coordinate agents. It should decide what each agent can
see, what it can change, how its output is audited, and when prior memory or
cached answers become stale.

## Core Outcome

Every user-facing answer, model call, and canonical write should be traceable to:

- the structured memory it used;
- the policy and taint class under which it was allowed;
- the object and namespace dependencies it relied on, including absences;
- the agent or route contract that authorized it;
- the provenance, ledger, and output events it produced;
- the invalidations it triggered.

## Domain-Neutral Kernel

The broker should expose neutral names even while the current database keeps
legal names:

| Current name | Neutral broker name |
| --- | --- |
| matter | workspace |
| document | artifact |
| span | span |
| actor | entity |
| assertion | claim |
| assertion_occurrence | claim_occurrence |
| assertion_link | claim_relation |
| issue | objective_node |
| issue_predicate | criterion |
| evidence_edge | support_edge |
| gap | gap |
| verification_state | review_state |
| provenance_event | provenance_event |
| ledger_event | run_event |
| llm_call | model_call |

Legal-specific enum values become one `DomainProfile`. Future profiles define
their own source roles, trust weights, claim kinds, objective templates,
relation types, policy rules, and output channels over the same kernel.

Every packet, output, write, cache key, policy decision, and run event binds
`domain_profile_id`, `domain_profile_version`, and `profile_mapping_hash`.
Changing a profile version invalidates packet and cache reuse for that profile.

## Design Loop So Far

### V1 Rejected

V1 proposed "Memory as Operating System" with a broker between every agent and
every store. Reviewer 1 rejected it as too broad and too abstract. The main
defects were negative dependencies, weak privilege handling, incomplete audit
manifests, vague write permissions, task-board drift, and a big-bang migration.

### V2 Rejected

V2 narrowed the broker to context assembly and writes. Reviewer 2 rejected it
because it still lacked enforceable boundaries: namespace absence was unsound,
internal guidance taint was under-modeled, direct query/trace routes were escape
hatches, packet storage was ambiguous, model-call selection was unaudited, write
commands had no transition contracts, and answerability states lacked an API
envelope.

### V3 Rejected

V3 added namespace freshness, packet/output taint, packet storage, brokered
writes, output events, and domain-neutral aliases. Reviewer 3 rejected it because
it still left unsafe legacy caches, lacked a model-call inventory gate, did not
define canonical dependency predicates, did not close freshness transactionally,
tracked taint only at packet/output level, left selector calls and raw output
paths as leak points, under-specified exemption lifecycle, and omitted domain
profile version binding.

### V4 Rejected

V4 added legacy cache containment, model-call registry, dependency predicate
schema, transactional CAS, object taint, central output emission, fail-closed
exemptions, profile version binding, and allowed legacy callers. Reviewer 4
rejected it because it still allowed fail-open migration gaps: incomplete
namespace coverage, permissive legacy cache reuse, taint labels without a
lattice, too-narrow model-call boundary, phase-only legacy caller expirations,
under-specified positive dependency validation, and no quarantine for objects
written under old domain profiles.

### V5 Rejected

V5 closed several escape hatches, but Reviewer 5 rejected it because the
namespace matrix still omitted implemented canonical surfaces, the required
namespace list and matrix disagreed, `legacy_untrusted` cache hits still had a
semantic planning path, `unknown_taint` was not fully quarantined, and
`DependencyManifest` lacked typed namespace dependencies plus enforcement
metadata.

### V6 Rejected

V6 added schema-table namespace coverage and a linter, but Reviewer 6 rejected
it because table coverage alone gave false confidence. The protocol still did
not model composed read surfaces such as `MatterModel.build_query_context`,
semantic cache stages still had active legacy influence in code, object-taint
coverage omitted read-influencing objects, and domain-profile remap lacked
revision namespaces and enforceable compatibility storage.

### V7 Rejected

V7 added a checked surface manifest, read/write surface linting, domain-profile
namespaces, expanded taint coverage, and semantic cache fail-closed behavior for
`orient`, `search_analysis`, and `synthesis`. Reviewer 7 rejected it because
`cascade_decision` was still an active unbrokered semantic route cache, and
semantic cache rows could self-attest `broker_validated` without a real broker
manifest record.

### V8 Rejected

V8 fixed the semantic-cache blocker, but Reviewer 8 rejected it because the
surface manifest still only checked declared surfaces. User-facing family
handlers and canonical writer methods could exist in code without manifest
coverage, expiry, taint behavior, or profile behavior.

### V9 Rejected

V9 added AST-discovered user-facing handlers and canonical writers, plus a
first broker substrate. Reviewer 9 rejected it because broker-store surfaces
were not attributed to their own namespaces, prefix discovery still missed real
mutators, and broker-store writes did not bump profile/taint namespaces
transactionally.

## V10 Architecture

V10 is an enforceable memory coordination protocol. It removes the remaining
fail-open escape hatches from V9.

It does not replace `MatterModel`. It adds a coordination layer over
model-call context, user-facing output audit, and canonical writes for migrated
paths.

Every agent action is classified as one of:

- `observe`
- `assemble_context`
- `call_model`
- `transform_output`
- `write_candidate`
- `review_transition`
- `user_steering`
- `maintenance`

Every action has:

- policy audience;
- taint class;
- dependency manifest;
- output or write contract.

## Immediate Legacy Cache Containment

The current code already uses `reasoning_cache` and synthesis response reuse.
Those caches must be treated as unsafe until brokered.

Before packet and answer caching are enabled:

- every existing cache stage must be registered in `cache_registry`;
- every cache entry must carry `taint_class`, `domain_profile_id`,
  `domain_profile_version`, `dependency_manifest_hash`, and `broker_status`;
- legacy entries without these fields are `legacy_untrusted`;
- `legacy_untrusted` cache hits may not influence clean user-facing text,
  answerability state, canonical writes, or routing that avoids required safety
  checks;
- `legacy_untrusted` may only drive non-semantic performance work such as
  warming a cache, loading files into memory, or prefetching artifacts whose
  inclusion will still be decided by clean brokered retrieval;
- `legacy_untrusted` may not affect candidate selection, retrieval terms,
  ranking, ordering, pruning, answerability, routing, packet composition,
  context ordering, write eligibility, or any model/tool call that can change
  semantic output;
- every cache hit must carry a revalidatable dependency manifest or a pointer to
  the packet/output event that carries it; a manifest hash alone is not enough;
- synthesis response cache must either be disabled for brokered outputs or
  wrapped so cached text cannot cross taint or audience boundaries.

This prevents old prompt-hash caches from becoming a hidden answer source after
the broker exists.

Current repair status: `ReasoningCacheStore` hard-wraps `cascade_decision`,
`orient`, `search_analysis`, and `synthesis` rows as `legacy_untrusted`. Reads
for those semantic stages return `None` until a real broker manifest store and
validator exist. Self-attested row JSON is never enough to reuse semantic cache.

## V11 Repair Slice

V11 makes the first broker enforcement path concrete instead of treating broker
state as inert metadata.

Current V11 implementation status:

- `memory_surface_manifest.json` has no broad writer namespace groups. Writer
  entries are flattened per surface and linted against table-derived namespaces.
- `tools/check_memory_namespace_coverage.py` derives write tables from mutating
  SQL and store delegation, maps them through the namespace matrix, and rejects
  extra or missing writer namespaces.
- `MemoryBrokerStore.answer_clarification_with_cas` is the first brokered CAS
  writer. It validates expected namespace revisions under `BEGIN IMMEDIATE`,
  requires a current domain profile, updates the clarification answer, records
  clean object taint, and bumps exact clarification/object-taint/policy
  namespaces in the same transaction.
- `MatterModel.answer_clarification` and the clarification-answer API route use
  that brokered writer, so the pilot path is exercised by the user-facing
  clarification workflow rather than only by tests.
- `MatterModel.build_query_context` filters tainted context rows for gaps,
  issues, clarifications, annotations, and assumptions. Legacy rows with no
  taint record remain visible during migration; rows with non-clean object
  taint are quarantined from clean query context.
- Regression tests prove overbroad writer declarations fail, stale CAS writes
  fail, and tainted answered clarifications are excluded from query context.

This is still a slice, not full migration: legacy direct writers remain
manifested and must be migrated or isolated namespace by namespace.

## V12 Repair Slice

V12 addresses the remaining V11 taint/profile blockers.

Current V12 implementation status:

- `MatterModel.build_query_context` filters every orientation field that carries
  canonical object identity: gaps, issues, actors, assertion-derived document
  ids, predicate hints, answered clarifications, annotations, assumptions, and
  the clean issue-coverage estimate used to choose `weakest_issue_id`.
- Assertion-derived context excludes rows tainted at the assertion,
  assertion-occurrence, artifact/document, or speaker-actor level.
- `object_taint` now stores `domain_profile_id`, `domain_profile_version`, and
  `profile_mapping_hash`; schema version 60 adds those columns for existing DBs.
- Brokered clarification writes require a compatible `profile_mapping`, CAS-check
  `profile_mappings:*`, `profile_mappings:profile:<profile_id>`, and
  `profile_mappings:mapping:<hash>`, and bind the resulting taint row to the
  exact profile version and mapping hash.
- Regression tests cover tainted actors, tainted document ids, tainted predicate
  hints, tainted clarifications, stale CAS writes, and profile metadata binding.

This remains a pilot path plus enforcement guardrails, not a claim that every
legacy writer has been migrated.

## Inference Call Registry

All inference and external-content processing calls must be inventoried before
enforcement begins. This includes chat/completion calls, embedding calls,
rerankers, OCR/document parsers, classifiers, tool-mediated model calls,
selectors, extractors, and any external service that consumes protected or
canonical state or influences output/write decisions.

`inference_call_registry` is a checked repo artifact. Each entry includes:

- usage label;
- owner module and call site;
- action kind;
- call modality: text generation, embedding, rerank, OCR/parser, classifier,
  selector, tool call, or external enrichment;
- whether the call can influence canonical memory;
- current status: `packetized`, `exempted`, `legacy_allowed`, or `blocked`;
- migration phase;
- allowed packet purposes;
- allowed taint classes;
- allowed output channels;
- whether canonical writes from this call are permitted.

CI/lint rules:

- any new inference/external-content call must reference a registry entry;
- calls that can influence canonical writes cannot be `legacy_allowed`;
- expired exemptions fail CI and fail runtime in strict mode.

## Memory Request

`MemoryRequest` describes what an agent wants from memory.

Required fields:

- `purpose`: `router_snapshot`, `read_synth`, `investigate_orientation`,
  `extraction`, `synthesis`, `deliverable`, `scenario`, `trace_audit`,
  `steer_preview`, or another registered purpose.
- `actor_id`, `actor_role`, and auth scopes.
- `audience`: internal, clean, export, court, opposing party, or a
  domain-profile-defined audience.
- `output_channel` and `persistence_target`.
- `disclosure_basis` and policy profile.
- optional target object: workspace, artifact, claim, objective node, run, or
  output.
- section intents: proof posture, source facts, candidate caveats, gaps,
  quants, entities, authorities, user guidance, citations.
- token budget and materiality floor.
- dependency mode: positive only, include negative namespaces, or strict absence
  tracking.

## Memory Packet

`MemoryPacket` is immutable once assembled.

It contains:

- packet id, broker version, request hash, packet hash;
- ordered sections with text, text hash, token estimate, materiality, policy
  status, and selector reason;
- positive object dependencies with kind, id, row digest or row version, belief
  state, verification state, and policy state at assembly time;
- negative dependencies with namespace, query predicate, and revision;
- namespace fingerprints consulted;
- policy decisions before inclusion;
- omitted material sections and reason: budget, policy, no data, stale,
  unimplemented;
- answerability recommendation;
- allowed citation objects;
- forbidden internal guidance objects.

## Namespace Freshness

V6 requires namespace freshness early, not as a distant optimization.

Add `namespace_revision`:

```text
namespace_revision(
  id,
  workspace_id,
  namespace,
  target_kind,
  target_id,
  revision,
  updated_at
)
```

Required namespaces:

- workspaces
- claims
- claim_occurrences
- claim_relations
- spans
- quants
- objective_nodes
- criteria
- support_edges
- proof_state
- review_state
- artifacts
- artifact_relations
- artifact_cards
- entities
- entity_aliases
- entity_affiliations
- document_actor_roles
- trust_overrides
- gaps
- policy
- content_policy
- object_taint
- annotations
- guidance
- authority
- assumptions
- namespace_revision
- domain_profiles
- profile_mappings
- clarifications
- revision_events
- belief_revisions
- propagation_queue
- cache_records
- provenance
- ledger
- runs
- inference_calls
- schema_state
- migrations

Every canonical table and read surface maps to one or more required namespaces.
There is no "where practical" exception for migrated paths.

`memory_surface_manifest.json` is the checked broker surface manifest. It names:

- composed read surfaces such as `MatterModel.build_query_context`;
- direct writer surfaces that still need broker migration;
- semantic cache stages that must miss without broker validation;
- object kinds requiring taint coverage;
- domain-profile revision namespaces;
- inference-call usage labels that must be in the registry.

### Namespace Coverage Matrix

This matrix must be generated or linted from `src/irys/matter/schema.py` plus
known read surfaces before enforcement. A canonical table missing from the
matrix is a broker adoption blocker.

Coverage lint checks schema tables, declared read surfaces, semantic cache
guards, domain-profile namespaces, object-taint coverage, and inference-call
registry coverage:

```powershell
python tools\check_memory_namespace_coverage.py
```

| Current table/read surface | Required namespace bumps |
| --- | --- |
| `matter` | `workspaces:*`, `workspaces:workspace:<matter_id>` |
| `assertion` | `claims:*`, `claims:claim:<id>` |
| `assertion_occurrence` | `claim_occurrences:*`, `claim_occurrences:claim:<claim_id>`, `artifacts:artifact:<artifact_id>` when source-linked, `spans:artifact:<artifact_id>` when span-linked |
| `assertion_link` | `claim_relations:*`, `claim_relations:claim:<source_id>`, `claim_relations:claim:<target_id>` |
| `belief_revision_event` | `belief_revisions:*`, `belief_revisions:claim:<assertion_id>`, `revision_events:*` |
| `assertion_revision` | `revision_events:*`, `revision_events:claim:<assertion_id>`, `claims:claim:<assertion_id>` |
| `run_session` | `runs:*`, `runs:run:<run_id>` |
| `ledger_event` | `ledger:*`, `ledger:<entity_kind>:<entity_id>` when linked |
| `document_inventory` | `artifacts:*`, `artifacts:artifact:<id>` |
| `document_relation` | `artifact_relations:*`, `artifact_relations:artifact:<source_doc_id>`, `artifact_relations:artifact:<target_doc_id>` |
| `document_card` | `artifact_cards:*`, `artifact_cards:artifact:<id>`, `policy:*` when privilege changes |
| `span` | `spans:*`, `spans:artifact:<artifact_id>` |
| `actor` | `entities:*`, `entities:entity:<id>` |
| `actor_alias` | `entity_aliases:*`, `entity_aliases:entity:<actor_id>`, `entities:entity:<actor_id>` |
| `actor_affiliation` | `entity_affiliations:*`, `entity_affiliations:entity:<actor_id>`, `entity_affiliations:entity:<org_actor_id>` |
| `document_actor_role` | `document_actor_roles:*`, `document_actor_roles:artifact:<doc_id>`, `document_actor_roles:entity:<actor_id>`, `artifacts:artifact:<doc_id>`, `entities:entity:<actor_id>` |
| `issue` | `objective_nodes:*`, `objective_nodes:objective_node:<id>` |
| `issue_predicate` | `criteria:*`, `criteria:objective_node:<issue_id>` |
| `assertion_issue_link` | `support_edges:*`, `support_edges:objective_node:<issue_id>`, `support_edges:claim:<claim_id>` |
| `gap` / `gap_link` | `gaps:*`, `gaps:<target_kind>:<target_id>` |
| `assumption` / `assumption_link` | `assumptions:*`, `assumptions:<target_kind>:<target_id>` |
| `evidence_link` | `support_edges:*`, `support_edges:<target_kind>:<target_id>`, `support_edges:artifact:<document_id>` when document-linked |
| `evidence_edge` | `support_edges:*`, `support_edges:<target_kind>:<target_id>`, `support_edges:<source_kind>:<source_id>` |
| `quant_fact` | `quants:*`, `quants:<target_kind>:<target_id>` when target-linked |
| `clarification_question` | `clarifications:*`, `clarifications:objective_node:<issue_id>` when linked, `guidance:*` when injected into reasoning |
| `reasoning_cache` | `cache_records:*`, `cache_records:cache:<cache_key>` |
| `document_trust_override` | `trust_overrides:*`, `trust_overrides:artifact:<document_pattern>`, `policy:*` |
| `document_annotation` | `annotations:*`, `annotations:artifact:<artifact_ref>`, `guidance:*` if injected into reasoning |
| `authority` / `authority_issue_link` | `authority:*`, `authority:<target_kind>:<target_id>` |
| `proof_state` | `proof_state:*`, `proof_state:objective_node:<issue_id>` |
| `decision_context` | `guidance:*`, `guidance:workspace:<workspace_id>` |
| `schema_version` | `schema_state:*`, `schema_state:workspace:<workspace_id>` |
| `pending_propagation` | `propagation_queue:*`, `propagation_queue:<queue>:<target_id>` |
| `llm_call` | `inference_calls:*`, `inference_calls:call:<id>` |
| `schema_migration` / `migration_backfill_job` | `migrations:*`, `migrations:job:<id>` |
| `verification_state` / `verification_event` | `review_state:*`, `review_state:<target_kind>:<target_id>` |
| `provenance_event` | `provenance:*`, `provenance:<target_kind>:<target_id>` |
| `content_policy_audit` / content policy decisions | `policy:*`, `content_policy:*`, `policy:<target_kind>:<target_id>` |
| `namespace_revision` | `namespace_revision:*`, `namespace_revision:<target_kind>:<target_id>`, `schema_state:*`, `schema_state:workspace:<workspace_id>` |
| `object_taint` | `object_taint:*`, `object_taint:<target_kind>:<target_id>`, `policy:*` |
| `domain_profile` | `domain_profiles:*`, `domain_profiles:profile:<domain_profile_id>` |
| `profile_mapping` | `profile_mappings:*`, `profile_mappings:profile:<domain_profile_id>`, `profile_mappings:mapping:<profile_mapping_hash>` |

Legacy canonical writes in a migrated namespace must either bump the required
namespace revisions and carry taint/provenance, or be invisible to brokered
clean packets.

Mutations bump both coarse and target-specific namespace revisions. Example:
writing a quant for objective node `X` bumps `quants:*` and
`quants:objective_node:X`.

Negative dependencies are first-class. If a packet says no quants existed for
objective `X`, it records the exact predicate and namespace revision used to
prove that absence.

## Dependency Predicate Schema

Negative dependency predicates must use a canonical schema:

```text
DependencyPredicate(
  namespace,
  target_kind,
  target_id,
  filters_json,
  projection,
  ordering,
  limit,
  policy_audience,
  domain_profile_id,
  domain_profile_version
)
```

Hashing rules:

- JSON is normalized with sorted keys and compact separators;
- omitted optional filters normalize to explicit null/default values;
- projection is explicit, even for existence checks;
- limit/cap is part of the predicate;
- policy audience and profile version are part of the predicate;
- when the predicate maps to SQL, a query-template id and normalized parameters
  are recorded so review can distinguish equivalent queries from similar prose.

During migration, namespaces without revision support must re-run their absence
query before model call and again before output/write commit.

## Dependency Manifest

Every packet, cache entry, output event, and brokered write carries a complete
`DependencyManifest`, not only hashes.

```text
DependencyManifest(
  manifest_schema_version,
  broker_version,
  workspace_id,
  domain_profile_id,
  domain_profile_version,
  profile_mapping_hash,
  positive_objects[],
  negative_predicates[],
  namespace_revisions[],
  policy_audience,
  taint_floor,
  validation_status,
  validation_results[],
  validator_version,
  validated_at,
  created_at
)
```

`validation_status` is one of `unvalidated`, `valid`, `stale`, `invalid`,
`blocked_by_unknown_taint`, or `blocked_by_policy`. `unvalidated` manifests
cannot be used for cache reuse, output emission, or brokered writes.

Namespace revision dependency:

```text
NamespaceRevisionDependency(
  namespace,
  target_kind,
  target_id,
  expected_revision,
  scope,
  predicate_id,
  required,
  observed_at
)
```

Rules:

- `namespace` must be one of the required namespaces in the coverage matrix;
- `target_kind` and `target_id` identify either the coarse `*` revision or the
  target-specific revision;
- `expected_revision` is the integer revision observed during packet assembly;
- `scope` is `positive_object`, `negative_predicate`, `policy`, `taint`,
  `profile_mapping`, `cache_record`, `output_transform`, or `write_guard`;
- `predicate_id` links absence claims to a `DependencyPredicate` when
  applicable;
- `required=false` is allowed only for diagnostic dependencies and can never
  satisfy a write or output guard.

Dependency validation result:

```text
DependencyValidationResult(
  dependency_kind,
  dependency_id,
  status,
  observed_revision,
  observed_digest,
  failure_reason,
  checked_at
)
```

Validators write one result per positive object, negative predicate, namespace
revision dependency, policy check, taint check, and profile mapping check.

Positive object dependency:

```text
PositiveObjectDependency(
  target_kind,
  target_id,
  namespace,
  row_digest,
  row_digest_algorithm,
  namespace_revision,
  taint_class,
  review_state,
  policy_state
)
```

Row digest rules:

- every canonical object used by brokered memory must expose either a
  monotonic row version or a canonical digest;
- digest JSON uses sorted keys, compact separators, and explicit nulls;
- volatile fields such as `updated_at` are excluded unless they affect meaning;
- semantic fields, review status, taint class, source role/trust, policy state,
  and profile id/version are included;
- validation re-reads the object and recomputes the digest.

Negative dependency validation re-runs the canonical predicate or checks that
the relevant namespace revisions have not changed. Cache/output reuse requires
full manifest validation. A manifest hash alone can identify a manifest, but it
cannot substitute for the manifest.

## Transactional Freshness CAS

Freshness validation and output/write commit must be transactionally closed.

For migrated writes:

1. Begin an immediate transaction.
2. Read expected namespace revisions from the packet dependency manifest.
3. Compare current revisions to expected revisions.
4. Abort with `stale_packet` on mismatch.
5. Perform canonical write.
6. Append provenance, ledger, and write result events.
7. Bump namespace revisions.
8. Commit.

For output events:

1. Begin an immediate transaction.
2. Compare packet dependency revisions and output dependency revisions.
3. Append `output_event` and transform events.
4. Commit.

No check-then-write gap is allowed for brokered paths.

## Taint Classes

The broker tracks taint on packets and outputs:

- `public_clean`: no privileged, internal, or non-citable content influenced
  the output.
- `clean_with_withheld`: clean output that includes withheld placeholders or
  gap indicators.
- `internal_work_product`: attorney guidance, strategy, privileged docs, or
  internal review notes influenced the output.
- `sealed_privileged`: raw privileged content entered the packet or prompt.

Outputs derived from `internal_work_product` or `sealed_privileged` must never
be reused for clean, export, court, or opposing-party surfaces. They need
separate caches and output records.

Policy is pre-inclusion. Post-redaction is only defense in depth.

## Taint Lattice And Object-Level Taint

Packet/output taint is not enough. Canonical objects also need taint metadata
and deterministic propagation.

Taint lattice, from least to most restrictive:

```text
public_clean
clean_with_withheld
internal_work_product
sealed_privileged
unknown_taint
```

Merge rule: derived taint is the maximum taint of all inputs. `unknown_taint`
is a quarantined top taint, not merely internal work product. Unknown-taint
objects are unusable for normal packet assembly, semantic planning, canonical
derivation, output emission, answer caching, cache reuse, or brokered writes
until classified or explicitly cleared by policy.

Add object-taint tracking for:

- artifacts;
- artifact cards;
- artifact relations;
- spans;
- entities;
- entity aliases;
- entity affiliations;
- document actor roles;
- claims;
- claim occurrences;
- claim relations;
- objective nodes;
- criteria;
- support edges;
- quant facts;
- authorities;
- gaps;
- annotations;
- guidance;
- assumptions;
- clarifications;
- revision events;
- belief revisions;
- propagation queue entries;
- trust overrides;
- generated work product;
- output transforms;
- cached answers.

Object taint records:

- target kind/id;
- taint class;
- source packet id or provenance event id;
- policy decision id;
- derivation reason;
- created timestamp.

Clean packets must consult object taint before including canonical objects. If a
claim was derived from privileged/internal material, a clean packet cannot use it
as clean evidence unless a later human/legal process explicitly clears or
re-sources it under the domain policy.

Internal agents may inspect unknown-taint objects only through quarantine
review commands. They may not use those objects as hints, ranking features,
context candidates, or derivation inputs for ordinary work.

Clearance workflow:

- only configured human/legal reviewer roles may clear or downgrade taint;
- clearance records basis, reviewer, scope, source replacement if any, and
  affected object ids;
- clearance bumps `policy:*` and the object namespace;
- old privileged provenance remains visible to internal audit even if clean
  reuse becomes allowed.

## Packet Storage

Use two storage surfaces:

### `memory_packet_event`

Stores the redacted audit manifest:

- request hash and redacted request JSON;
- section ids and section hashes;
- object ids;
- dependency manifest;
- policy manifest;
- omitted manifest;
- taint class;
- answerability state;
- packet hash;
- run id;
- optional model call id;
- created timestamp.

### `memory_packet_blob`

Stores sealed raw packet text only when retention policy allows it. Clean/export
readers cannot access this store. The storage layer must enforce access control
or encryption before privileged packet retention is enabled.

Prompt hash binds packet hash, system prompt hash, and model call config. Output
hash is recorded after the model call. `llm_call_id` links them.

## Model-Call Rule

All text-bearing model calls must be packetized or explicitly exempted.

Exemptions require `model_call_exemption_event` with reason, owner, risk class,
and allowed duration. No canonical write may be based on an exempted call unless
that output is later converted into brokered candidate writes with provenance
noting the exemption.

Exemption lifecycle:

- every exemption has `expires_at`;
- expired exemptions fail CI;
- expired exemptions fail runtime in strict mode;
- exemption renewal requires a new owner and reason;
- exempted model output cannot directly write canonical memory;
- conversion from exempted output to canonical candidate memory must go through
  brokered write commands and mark provenance as exemption-derived.

Section selection should be deterministic by default. If model-based selection
is used, the selector call itself gets a packet and writes selector input/output
hashes.

Selector input sections must already be policy-filtered and taint-labeled before
the selector sees text.

## User-Facing Output Event

Every user-facing response writes `output_event`, including direct query and
trace routes.

Fields:

- output id;
- run id;
- route family and terminal family;
- answer state;
- taint class;
- policy audience;
- output channel;
- packet ids used;
- dependency manifest;
- citation manifest;
- withheld and omitted counts;
- transform chain hashes;
- response hash;
- created timestamp.

Direct DB routes do not need full packet text, but they must still record
dependencies, absence queries, policy decisions, and response hash.

## Central Output Emitter

No user-facing route should return raw strings or ad hoc result objects.

All routes must call:

```text
emit_output(envelope, output_context)
```

`emit_output` is responsible for:

- validating packet/output dependency freshness;
- writing `output_event`;
- applying final policy checks;
- binding route audit;
- binding transform chain;
- returning the standardized answer envelope.

Allowed legacy response paths must be listed in `allowed_legacy_callers` with an
expiry and migration owner.

## Standard Answer Envelope

Every API/UI route should return:

- `answer_text`
- `answer_state`: `answerable`, `partial`, `withheld`, `stale`,
  `insufficient_record`, `requires_review`, or `policy_blocked`
- `taint_class`
- `citation_status`: `verified`, `mixed`, `candidate_only`, `none`, or
  `withheld`
- `used_packet_ids`
- `omitted_sections`
- `withheld_counts`
- `stale_reasons`
- `recommended_actions`
- route audit

The UI can render this compactly, but the backend should always carry the
envelope.

## Brokered Write Commands

Migrate only three write commands first:

1. `create_candidate_claim`
2. `create_candidate_support_edge`
3. `stage_user_correction`

Each command requires:

- actor id and role;
- run id;
- packet id or exemption id;
- idempotency key;
- target refs;
- expected current state where applicable;
- provenance context;
- policy and taint class.

Each command transaction:

1. Validate packet freshness.
2. Check actor role and command permission.
3. Check state transition.
4. Write canonical rows.
5. Write verification candidate or review task if applicable.
6. Append provenance event.
7. Append ledger event.
8. Bump namespace revisions.
9. Mark proof state stale or recompute bounded target.
10. Return write result.

Failure modes:

- `stale_packet`
- `policy_forbidden`
- `invalid_transition`
- `idempotency_conflict`
- `target_not_found`
- `privilege_taint_block`
- `proof_recompute_deferred`

Other direct `MatterModel` writes remain legacy. They must be listed in an
allowed legacy caller registry and burned down over time.

## Allowed Legacy Callers

`allowed_legacy_callers` is a checked artifact. Each entry includes:

- file/function;
- reason it is still legacy;
- whether it is user-facing;
- whether it can influence canonical memory;
- whether it can access privileged/internal material;
- migration owner;
- hard expiry date.

New direct canonical mutations outside brokered commands are blocked by tests or
lint once the command's migration phase starts.

Phase-only exemptions are forbidden. A legacy caller that can influence
canonical memory or access privileged/internal material must have runtime
blocking semantics after its expiry date. Expired entries fail CI and strict
runtime mode.

## Domain Profile Remap And Quarantine

Domain profile changes can alter semantic meaning. Existing objects written
under an older profile version must not silently remain clean-valid.

Domain profile state has explicit revision namespaces:

- `domain_profiles:*`
- `domain_profiles:profile:<domain_profile_id>`
- `profile_mappings:*`
- `profile_mappings:profile:<domain_profile_id>`
- `profile_mappings:mapping:<profile_mapping_hash>`

Each canonical object carries or can resolve:

- `domain_profile_id`;
- `domain_profile_version`;
- `profile_mapping_hash`;
- `remap_status`: `current`, `needs_remap`, `remapped`, `quarantined`, or
  `legacy_unknown`.

Compatibility rules are stored as typed records:

```text
ProfileCompatibilityRule(
  rule_id,
  source_domain_profile_id,
  source_domain_profile_version,
  target_domain_profile_id,
  target_domain_profile_version,
  source_mapping_hash,
  target_mapping_hash,
  target_kind,
  target_namespace,
  compatibility_status,
  required_transform_id,
  reviewer_id,
  created_at
)
```

`compatibility_status` is `compatible`, `requires_transform`, or
`incompatible`. Missing compatibility rules fail closed as `quarantined` for
clean packets and brokered writes.

When a profile changes:

- bump `domain_profiles:*`, `domain_profiles:profile:<domain_profile_id>`,
  `profile_mappings:*`, and the affected profile-mapping target revision;
- packet/cache reuse under the old profile is invalidated;
- affected objects move to `needs_remap` or `quarantined`;
- clean broker packets exclude `needs_remap`, `quarantined`, and
  `legacy_unknown` objects unless a domain-specific compatibility rule permits
  them;
- remap creates an auditable event linking old semantics to new semantics.

CAS guard: output emission and brokered writes compare the packet's
`domain_profiles` and `profile_mappings` namespace revisions before commit.
Any profile or mapping revision bump after packet assembly returns
`stale_profile_mapping` and forces reassembly or quarantine review.

## Output Transform Chain

Every post-model mutation is an `output_transform_event`.

Examples:

- advocacy gate insertion;
- quant supplementation;
- redaction;
- citation verification note insertion;
- formatting.

Each transform records kind, reason, before hash, after hash, policy decision
ids, actor or system, created timestamp, and any extra dependency manifest.

## Task Model

Start with `TaskView`, not a new source of truth.

`TaskView` derives prioritized work from:

- gaps;
- review queue;
- stale proof;
- missing artifacts;
- unanswered clarifications;
- stale packets;
- policy-blocked outputs.

Materialized tasks come later and must have leases, idempotency keys,
canonical-object refs, and reconciliation from canonical state.

## Migration Enforcement

Add enforcement tests:

- no new text-bearing model call without packet or exemption event;
- no new inference/external-content call without registry entry;
- migrated commands cannot call direct table writers outside broker;
- user-facing routes produce output event and answer envelope;
- clean outputs cannot use internal-work-product packets;
- negative dependencies invalidate on namespace revision bump;
- allowed legacy caller list is explicit and shrinking.
- inference call registry covers every model/tool/parser call that consumes
  protected/canonical state or affects output/write decisions;
- existing cache stages are registered and taint-contained;
- domain profile version participates in packet/output/cache keys;
- brokered writes use transactional namespace-revision CAS;
- object-level taint lattice blocks clean reuse of internal/privileged/unknown
  objects;
- clean broker packets exclude profile-stale/quarantined objects;
- cache reuse requires full dependency manifest validation;
- migrated namespace legacy writes either bump revisions and taint/provenance or
  remain invisible to brokered clean packets;
- all user-facing outputs go through `emit_output`.

## Adoption Sequence

1. Add `inference_call_registry`, `cache_registry`, and `allowed_legacy_callers`
   with hard expirations.
2. Contain existing cache stages as `legacy_untrusted` or disable unsafe
   synthesis cache reuse.
3. Add neutral broker dataclasses, domain profile ids/versions, and answer
   envelope types.
4. Add `namespace_revision` for the full namespace coverage matrix.
5. Add canonical dependency predicate and dependency manifest helpers.
6. Add `memory_packet_event`, `output_event`, object taint, and transform-event
   stores.
7. Add central `emit_output`.
8. Move read-family context to broker and return answer envelope.
9. Add output events to query and trace direct routes.
10. Add brokered `create_candidate_claim` and `create_candidate_support_edge`
   for one extraction path.
11. Add `stage_user_correction`.
12. Move synthesis packet assembly behind broker sections.
13. Add output transform events.
14. Only then add packet and answer caches keyed by dependency manifest and
    taint class.

## Open Review Status

Reviewer 7 rejected V7 because `cascade_decision` remained an active
unbrokered route cache and semantic cache metadata was self-attested by cache
row JSON rather than validated against a real broker manifest record.

Reviewer 8 rejected V8 because read/write surface coverage was still based on
declared manifest rows rather than discovered code surfaces.

Reviewer 9 rejected V9 because broker-store surfaces were grouped under a broad
graph manifest without their own namespaces, discovery still missed real
mutators such as clarification answers, document hash updates, predicate
resolution, proof-state computation, and assertion correction, and broker-store
writes did not bump their own namespace revisions.

Reviewer 10 rejected V10 because writer manifest groups were still overbroad,
broker freshness had revision increments but no compare-then-write CAS path,
and object-taint/domain-profile state was recorded but not enforced by a read
or write path.

V11 addresses the first repair slice by replacing broad writer groups with
exact per-surface writer declarations, adding linter rejection for overbroad
writer namespaces, adding `MemoryBrokerStore.answer_clarification_with_cas` and
`MatterModel.answer_clarification` as the first brokered CAS clarification
path, and filtering non-clean object-taint rows from query context. Current
lint coverage is 12 read surfaces, 97 write surfaces, 47
schema tables, 41 namespaces, and 21 inference labels. It is ready for fresh
review.

Reviewer 11 rejected V11 because tainted actors, assertion-derived document
ids, predicate hints, and coverage-derived issue focus could still leak into
orientation, and because the clarification CAS path required a current domain
profile but did not bind or CAS-check a profile mapping.

V12 addresses those blockers by extending query-context taint filtering to all
orientation fields and the clean coverage estimate, adding profile/version/hash
columns to `object_taint`, and requiring/binding compatible profile mappings in
the brokered clarification CAS path. Current lint coverage is 12 read surfaces,
97 write surfaces, 47 schema tables, 41 namespaces, and 21 inference labels. It
is ready for fresh review.

Reviewer 12 rejected V12 because tainted child issues could still become
`weakest_issue_id` through raw subtree rollup, and because brokered
clarification writes could accept a caller-supplied `profile_mapping_hash` that
did not match the compatible mapping row.

V13 addresses those blockers by selecting weakest leaves through a taint-aware
subtree path and by deriving the effective profile mapping hash from the
compatible mapping row. The clean weakness path excludes tainted issues,
criteria/predicates, support edges, assertions, occurrences, artifacts, and
actors before choosing `weakest_issue_id`. A caller-supplied mapping hash must
match the compatible mapping row or the broker rejects the write.
`MatterModel.answer_clarification` now resolves the current profile and
compatible mapping hash from broker state for the requested domain profile
instead of hardcoding a legal-only value. Current lint coverage is 12 read
surfaces, 97 write surfaces, 47 schema tables, 41 namespaces, and 21 inference
labels. It is ready for fresh review.

Reviewer 13 rejected V13 because `MatterModel.answer_clarification` could still
create or overwrite a profile/mapping when a domain profile existed but lacked a
compatible clarification mapping. That was not fail-closed for non-legal domain
profiles. The reviewer also noted that cross-domain brokered writes needed to
CAS-check the source profile revision, not just the target profile.

V14 addresses those blockers by making clarification answering require an
already-current domain profile and already-compatible mapping. The answer path
does not create or overwrite domain profiles or mappings. Cross-domain brokered
clarification writes require the source profile to exist/currently match and add
the source profile revision to CAS expectations. Current lint coverage is 12
read surfaces, 97 write surfaces, 47 schema tables, 41 namespaces, and 21
inference labels. It is ready for fresh review.
