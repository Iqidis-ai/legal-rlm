# Ideal Product Spec

This document is the durable product-level north star for this repository.

It is intentionally long and repetitive in places. Future sessions will not
have the benefit of the conversations that produced it. The point of this file
is to preserve the reasoning, the failure modes, the priorities, and the
desired behavior in one place so future work can start from the right frame.

This is not an implementation plan and not a sprint plan. It is an ideal-state
system specification for what this product should become.

## How To Use This Document

If you are a future session, a new contributor, or even the same contributor
coming back later, read this file as follows:

1. Read `Sections 1-6` to understand what product is actually being built and
   what it is explicitly not.
2. Read `Sections 7-12` to understand why the current style of system is not
   enough and where it tends to fail.
3. Read `Sections 13-27` to understand the required system models, memory
   layers, user interaction behavior, and operational loops.
4. Read `Sections 28-36` to understand the priority order, what matters most,
   what can come later, and what success actually looks like.

If there is ever tension between short-term product ideas and this document,
default to the logic in this file unless the product direction has been
explicitly changed.

## 1. What Product We Are Actually Building

We are not building "an LLM that answers legal questions about a folder."

We are not building "a fancy legal chatbot."

We are not building "a memo generator that happens to cite documents."

We are building a legal intelligence system that continuously constructs,
maintains, and improves a living model of a matter from messy, adversarial,
incomplete, evolving inputs.

That system should:

- know what exists in the matter and what is missing;
- know who said what, in what role, and for what likely purpose;
- know which statements are operative, which are disputed, which are strategic,
  and which are weak;
- know what likely happened, while keeping that separate from what can actually
  be proved;
- know what law, procedure, and practical context matter;
- know what the user is trying to accomplish right now;
- make its reasoning visible enough that users can follow, interrupt, redirect,
  and correct it while it is working;
- improve the matter model as users interact with it;
- reuse durable structure instead of recomputing everything from scratch;
- eventually produce not only text, but also structured analysis, quantitative
  outputs, and visual work product from the same underlying matter model.

The core product is not the answer text. The core product is the matter model.

## 2. Why This Matters

Legal work is not just about retrieving text and summarizing it. It involves:

- adversarial records;
- advocacy-heavy source material;
- incomplete repositories;
- shifting operative documents;
- contradictions across time and across sides;
- missing metadata;
- practical context that is not visible in the documents;
- quantitative relationships that matter as much as prose;
- strategic decisions that depend on who the decision-maker is and what they
  care about;
- ongoing user refinement and correction;
- very high compute budgets where waste compounds rapidly.

If the system only retrieves text, extracts claims, and drafts prose, it will
look coherent while still being deeply wrong, strategically naive, or
operationally wasteful.

The product must therefore be built around durable legal understanding, not
around one-shot response generation.

## 3. Why This File Exists

This file exists because future sessions will not carry the full conversational
context that led to these conclusions. Without a persisted spec, the system
direction will drift toward whatever is easiest to code, easiest to demo, or
easiest to explain, which is usually not what the product actually needs.

This document is meant to prevent that drift.

Specifically, it should help future sessions avoid:

- collapsing the product into "better prompts";
- optimizing for polished output before the underlying intelligence substrate is
  correct;
- treating more recursion or more token spend as a substitute for better state;
- confusing observability with logs instead of user-steerable reasoning state;
- confusing legal correctness with practical usefulness;
- forgetting that source identity, agenda, and procedural posture are part of
  the reasoning problem;
- rebuilding knowledge every query instead of maintaining it.

## 4. The Short Version

If this entire document had to be compressed into one paragraph, it would be
this:

The ideal product is a continuously maintained legal intelligence substrate for
each matter, with durable repository memory, document identity memory,
actor/contact memory, assertion and evidence memory, issue and authority memory,
decision-context overlays, explicit uncertainty, user-visible reasoning,
interruptible execution, incremental belief revision, and enough observability
to know where the model is weak, what changed, what is missing, and what should
happen next. Text answers, charts, timelines, research packets, and other work
product should all be downstream views over that substrate.

## 5. Product-Level Objectives

The product should satisfy all of the following at the same time.

### 5.1 Build A Durable Matter Model

The system should build and maintain a reusable model of the matter rather than
rediscovering the same structure on every query.

### 5.2 Stay Honest Under Uncertainty

The system should remain useful even when authorship is unclear, scans are bad,
document roles are uncertain, dates conflict, or key materials are missing.

### 5.3 Be User-Steerable In Real Time

The system should show enough of what it is doing that a user can stop it,
annotate it, redirect it, provide context, or tell it that a line of reasoning
is strategically irrelevant.

### 5.4 Improve From Interaction

User corrections should not vanish into chat history. They should improve the
matter model and improve future prioritization.

### 5.5 Support Real Legal Work

The system must help with record understanding, proof assessment, legal
analysis, strategic prioritization, research, and quantitative work. It cannot
be only a drafting engine.

### 5.6 Use Compute Intelligently

The system should spend expensive compute once on durable structure and reuse it
aggressively. Wasteful recompute is not just inefficient. At scale, it is a
product failure.

## 6. Non-Goals For The Current Priority Window

The following things may matter later, but they are not the center of gravity
right now:

- advanced branching and merging collaboration workflows;
- polished visual design as a standalone objective;
- broad generic assistant behavior unrelated to matter intelligence;
- shallow feature breadth at the expense of model quality;
- optimizing memo style before fixing matter understanding.

This does not mean they are unimportant forever. It means they are downstream of
the intelligence substrate and should not displace it.

## 7. The Fundamental Product Mistake To Avoid

The most dangerous mistake is to think the product is "good legal analysis."

That is still too shallow.

The system must separately model:

- what the documents say;
- what the documents are;
- who produced them;
- what role each source is playing;
- what likely happened in the world;
- what can actually be supported or proved;
- what the law says about the issue;
- what the user is trying to do;
- what the relevant decision-maker will care about;
- what remains uncertain or missing;
- what should be done next to improve the model.

If those layers are collapsed into one stream of "analysis," the system will
sound intelligent while quietly becoming self-reinforcing, circular, and
strategically brittle.

## 8. Product Failure Modes We Are Explicitly Designing Against

These are not edge cases. These are core failure modes of systems like this.

### 8.1 Recompute Waste

The system learns something important, fails to preserve it properly, and then
rediscovers it later at high cost.

This happens when:

- facts are stored as flat strings instead of structured objects;
- assertions are not linked to dependencies;
- document profiling is not persisted;
- issue understanding is query-local instead of matter-global;
- user corrections are not converted into durable state.

### 8.2 Circular Reasoning

The system sees a claim in a document, then sees arguments around that claim,
then sees later summaries that restate it, and gradually begins treating the
claim as ground truth.

This happens because:

- source roles are not modeled strongly enough;
- advocacy is not separated from operative content;
- repeated rhetoric gets mistaken for corroboration;
- memo text and extracted facts contaminate each other;
- there is no disciplined support-vs-attack structure for assertions.

### 8.3 Over-Amplification Of Lawyer-Written Material

Legal corpora contain a lot of adversarial prose. A system that is strong at
language but weak at source calibration will often over-weight whatever has the
cleanest argumentative form.

That means it may over-trust:

- pleadings;
- briefs;
- demand letters;
- declarations;
- lawyer summaries;
- internal explanatory memos.

### 8.4 Flat Matter Understanding

The system may extract lots of local facts but still fail to model:

- supersession;
- conditionality;
- exception structures;
- waiver;
- causation;
- chronology;
- version shifts;
- contradictions;
- proof gaps;
- issue-element coverage.

### 8.5 No Working Mental Model

A good lawyer updates their model as they learn. They do not just accumulate
notes. A system that only appends fact strings is not maintaining a mental
model. It is dumping observations without revision.

### 8.6 Strategic Naivete

The system may produce a legally correct analysis that is practically weak
because it ignores:

- judge preferences;
- venue realities;
- client goals;
- partner instructions;
- budget constraints;
- timing constraints;
- settlement posture;
- law firm strategy.

### 8.7 Hidden Assumptions

The system may rely on a fragile assumption but never surface it. Users then
cannot correct it, and downstream work product becomes quietly skewed.

### 8.8 Poor Missingness Detection

The system may fail to realize that absence itself matters.

Examples:

- missing signed amendment;
- missing notice;
- missing approval;
- missing payment proof;
- missing attachment;
- missing referenced schedule;
- missing authority on a required element.

### 8.9 Inadequate User Observability

If the user cannot see why the system is doing what it is doing, they cannot
steer it effectively, and the system will waste compute following the wrong
path.

### 8.10 Weak Quantitative Reasoning

Many legal questions are partially numeric. If numbers stay trapped in prose,
the system cannot do damages, reconciliation, trend analysis, or scenario
planning well.

### 8.11 Research As A Sidecar Instead Of A Layer

If legal research is handled as an afterthought, the system will not know when a
matter issue is blocked by law rather than facts, and authorities will remain
detached from issue reasoning.

### 8.12 No Durable Learning From User Input

If the user corrects the system and that correction does not materially change
future behavior, the system remains expensive and brittle.

## 9. The Current-System Diagnosis This Spec Is Responding To

This repository, in its current form, is closer to a recursive search,
read, and synthesis engine than a full legal intelligence platform.

At a high level, the current style of system tends to:

- search for text matches;
- analyze returned snippets or documents;
- append facts and citations;
- produce synthesized prose;
- do only limited structured revision of what it has learned.

That is a reasonable early architecture for experimentation, but it is not the
right final architecture for production legal work.

The main deficiencies of that style are:

- durable matter memory is too weak;
- document identity and source posture are too thin;
- facts are too flat and too append-only;
- user interaction does not sufficiently alter persistent intelligence;
- strategic overlays are not first-class;
- issue coverage is not the backbone of the system;
- repository intelligence is underdeveloped;
- quantitative structure is underdeveloped;
- observability is not rich enough for users to steer the run.

This spec is the answer to those deficiencies.

## 10. The Core Product Thesis

The central thesis of this product is:

Every expensive computation should either:

- improve the durable matter model; or
- read from the durable matter model to generate a user-facing artifact.

If a computation does neither, it is likely waste.

This principle should govern architecture, prioritization, and evaluation.

## 11. The Five Questions The System Must Never Confuse

The system must keep the following questions separate, even though they interact:

1. What does the record literally say?
2. What likely happened in the world?
3. What can be supported or proved from the available record?
4. What is the best legal analysis?
5. What will the relevant decision-maker actually care about?

Many system errors come from collapsing these layers.

Examples:

- A complaint allegation is in the record, but it is not necessarily what
  happened.
- Something may likely have happened, but still be weakly provable.
- An argument may be doctrinally strong, but strategically weak before a given
  judge.
- A client objective may make a "best legal argument" the wrong operational
  recommendation.

## 12. The Product Must Maintain Multiple Parallel Models

One global flat state is not enough. The system needs several linked models.

### 12.1 Repository Model

The repository model answers:

- what documents exist;
- what is missing;
- what is duplicate;
- what belongs together;
- what changed;
- what has been reviewed;
- what high-value areas remain unexplored.

This is the system's operational map of the corpus itself.

### 12.2 Record Model

The record model captures the literal documentary record:

- document contents;
- document structure;
- source spans;
- document types;
- speech acts;
- sender/recipient/signatory relationships;
- references between documents.

This is the closest layer to the actual materials.

### 12.3 Reality Model

The reality model is the system's best evolving view of what likely happened in
the world. It should be clearly marked as an inference layer, not identical to
the record.

### 12.4 Proof Model

The proof model captures what is actually supported, corroborated, contested,
missing, vulnerable, or strategically weak from an evidentiary perspective.

### 12.5 Legal Model

The legal model captures:

- claims and defenses;
- elements and sub-elements;
- contract questions;
- burdens;
- authorities;
- procedural posture;
- legal dependencies.

### 12.6 Decision-Context Model

This model captures practical context not reducible to legal doctrine:

- judge tendencies;
- partner preferences;
- firm playbooks;
- client goals;
- negotiation posture;
- budget limits;
- business constraints;
- timing constraints.

### 12.7 Interaction Model

The interaction model tracks:

- user objective;
- user corrections;
- user steering;
- current work mode;
- preferred output shape;
- urgency level.

### 12.8 Attention Model

This model decides where the system should spend more compute next by weighing:

- materiality;
- uncertainty;
- expected value of clarification;
- expected value of more retrieval;
- expected value of more synthesis;
- importance of unresolved issues;
- risk of stale assumptions.

## 13. Canonical Persistent Stores

These stores do not prescribe database technology. They describe what durable
state the product must preserve.

### 13.1 Matter Registry

The matter registry should hold the top-level identity and status of a matter,
including:

- matter ID;
- matter name;
- parties;
- key related entities;
- forum or venue;
- procedural posture;
- likely governing law;
- current objectives;
- known deadlines;
- major open issues;
- maturity status of the matter model.

### 13.2 Repository Inventory Store

This store should answer "what is in the repo" without re-scanning everything
from scratch every time.

It should include:

- stable document identities;
- hashes;
- path or storage references;
- ingest status;
- duplicate and near-duplicate relationships;
- family membership;
- version membership;
- processing status;
- unread/read coverage;
- likely salience;
- likely missing companion documents.

### 13.3 Document Card Store

Every document should have a durable card. That card should outlive any one
query.

Each document card should ideally include:

- canonical title;
- original filename;
- document type;
- likely subtype;
- source side;
- author;
- sender;
- recipient;
- signatories;
- creation date;
- sent date;
- effective date;
- discovery or ingestion date;
- likely purpose;
- likely rhetorical posture;
- likely reliability posture;
- privilege/confidentiality flags if known;
- operative status;
- version lineage;
- attachment chain;
- family context;
- issue links;
- importance or salience by issue;
- unresolved metadata flags.

### 13.4 Span Store

Every important assertion, number, event, visual point, or recommendation should
trace back to exact spans.

The span store should support:

- page spans;
- paragraph spans;
- line spans;
- section spans;
- clause spans;
- table spans;
- cell spans;
- exhibit references;
- attachment references.

This is the grounding layer that keeps the system honest.

### 13.5 Actor / Contact Store

This is more important than it first appears.

The system must know:

- who people and organizations are;
- which aliases refer to the same actor;
- which email addresses belong to whom;
- job titles and role changes over time;
- which law firms represent which parties;
- who tends to communicate for whom;
- who is adverse, aligned, neutral, or unknown;
- what recurring incentives or agendas may exist.

This store enables:

- source-role reasoning;
- communication mapping;
- speaker identification;
- authorship inference;
- agenda estimation;
- better retrieval planning.

### 13.6 Assertion Store

The assertion store is the heart of the intelligence layer.

The atomic unit should be a typed assertion, not a flat fact string.

Each assertion should ideally capture:

- normalized proposition;
- source span links;
- source document links;
- speaker;
- source role;
- source side;
- speech-act classification;
- temporal scope;
- issue links;
- dependencies;
- support links;
- attack links;
- status;
- uncertainty;
- provenance of origin;
- whether it is extracted, inferred, imported, or user-supplied.

Assertions should support statuses like:

- `alleged`;
- `argued`;
- `admitted`;
- `testified`;
- `operative`;
- `performed`;
- `not performed`;
- `disputed`;
- `superseded`;
- `withdrawn`;
- `inferred`;
- `resolved`;
- `unknown`.

### 13.7 Evidence Store

Evidence is not identical to assertions.

The evidence store should capture:

- support relationships;
- contradiction relationships;
- corroboration relationships;
- authentication notes;
- proof strength;
- source diversity;
- vulnerabilities;
- issue-element relevance;
- admissibility or practical proof concerns where known.

### 13.8 Issue Store

The issue store should capture structured legal and matter questions.

It should support:

- claims;
- defenses;
- sub-issues;
- legal elements;
- contract interpretation questions;
- factual predicates;
- procedural gates;
- damages components;
- burden allocations;
- materiality scores;
- practical salience scores;
- support and attack sets;
- unresolved predicates.

### 13.9 Assumption Store

The assumption store is critical because assumptions are where quiet failure
often enters.

Each assumption should capture:

- the assumption itself;
- why it exists;
- what evidence supports it, if any;
- what evidence would invalidate it;
- what downstream conclusions depend on it;
- whether it came from the system or a user;
- whether it is temporary, provisional, or accepted for current work.

### 13.10 Gap Store

This should preserve all known missingness, not just generic TODOs.

Examples:

- missing documents;
- missing metadata;
- missing issue predicates;
- missing authorities;
- missing user context;
- missing quantitative inputs;
- unresolved contradictions;
- expected but absent attachments;
- expected but absent notices.

### 13.11 Quant Store

The quant store should hold normalized numeric structure.

It should contain:

- amounts;
- currencies;
- dates;
- date ranges;
- rates;
- balances;
- invoices;
- payment events;
- ledger mappings;
- formulas;
- scenario variables;
- damages components;
- reconciliations;
- conflicts.

### 13.12 Authority Store

Authorities must be stored as first-class objects, not floating citations.

The authority store should include:

- jurisdiction;
- date;
- source type;
- holding or rule;
- scope;
- treatment;
- procedural posture;
- issue links;
- supportive role;
- adverse role;
- limiting role;
- freshness metadata.

### 13.13 Decision-Context Store

This store should hold practical overlays such as:

- judge comments or preferences;
- forum tendencies;
- partner instructions;
- client goals;
- client risk tolerance;
- budget constraints;
- business constraints;
- negotiation objectives;
- presentational preferences.

This layer must influence prioritization and recommendation, but it must not
rewrite the canonical record or truth layers.

### 13.14 Work-Product Store

The system should preserve reusable derived artifacts like:

- timelines;
- issue boards;
- contradiction registers;
- research packets;
- damages schedules;
- open-question lists;
- next-step lists;
- memo scaffolds;
- evidence matrices;
- obligation trackers.

### 13.15 Reasoning Ledger

This is not raw hidden chain-of-thought. It is a structured, user-facing ledger
of what the system did and why.

It should log:

- current objective;
- active branch;
- recent evidence changes;
- active assumptions;
- current uncertainties;
- why a new step was chosen;
- what changed the model;
- what the system plans to do next.

## 14. Repository Intelligence Requirements

The system should always know the repository inventory.

That sounds basic, but it is one of the most important capabilities. The system
cannot reason efficiently if it does not first know what exists, what does not
exist, and what belongs together.

Repository intelligence should support:

- document inventory without expensive rediscovery;
- document family construction;
- version-chain detection;
- duplicate and near-duplicate clustering;
- issue-to-document coverage tracking;
- read coverage tracking;
- high-salience unread region detection;
- likely missing companion document detection;
- negative knowledge, meaning what was searched and found not to be useful.

The system should know that "what to read next" is often a repository problem
before it is a reasoning problem.

## 15. Document Identity, Role, And Agenda Intelligence

The system must know more about a document than just its text.

For each important document, it should try to answer:

- What kind of document is this?
- What role is it playing in the matter?
- Who authored it?
- Who signed it?
- Who received it?
- Which side is it associated with?
- What is its likely purpose?
- Is it likely operative, rhetorical, explanatory, evidentiary, or procedural?
- What are the likely distortions or limitations of this source type?

This matters because a signed amendment, a demand letter, a complaint, an
internal spreadsheet, and a judge order may all mention the same proposition but
mean very different things.

The system should have a disciplined way of reasoning about source posture:

- advocacy source;
- operational business source;
- neutral procedural source;
- authoritative decision-maker source;
- informal communication source;
- draft or negotiation source;
- post hoc explanatory source.

It should not claim certainty about agenda when it does not have certainty, but
it should preserve structured best estimates and explain why they matter.

## 16. Actor And Contact Intelligence

Actor memory is not optional.

The system should maintain durable knowledge about:

- people;
- organizations;
- aliases;
- email identities;
- changing titles;
- firm and party affiliations;
- who speaks for whom;
- recurring communication patterns;
- recurring reliability concerns;
- recurring strategic incentives.

Why this matters:

- If the system knows who is speaking, it can better calibrate trust.
- If it knows who regularly communicates about a topic, it can retrieve more
  efficiently.
- If it knows who is counsel versus business personnel versus neutral third
  party, it can interpret statements differently.
- If it knows that multiple names or addresses map to one actor, it can avoid
  fragmented reasoning.

The system should not only know "John Doe authored this email." It should know
whether John Doe is outside counsel, GC, CFO, negotiator, expert, board member,
or counterparty contact, and how that role changes how the email should be
interpreted.

## 17. Assertions, Speech Acts, And Truth Maintenance

The system should not flatten everything into "facts."

It should distinguish speech acts such as:

- alleged;
- argued;
- denied;
- admitted;
- ordered;
- performed;
- paid;
- requested;
- threatened;
- promised;
- estimated;
- calculated;
- observed;
- testified;
- stipulated;
- amended;
- waived;
- terminated.

This distinction matters because the same proposition can occupy different roles
depending on who is saying it and in what document.

The system should maintain belief revision over time:

- later evidence can supersede earlier evidence;
- later operative documents can replace earlier operative documents;
- a court order can narrow what matters procedurally;
- user correction can invalidate a prior assumption;
- contradictory evidence can force multiple coexisting live theories.

The critical point is this:

The product needs a truth-maintenance system, not an append-only notes system.

## 18. The Difference Between Assertions, Evidence, And Conclusions

These are not the same thing and should never be treated as the same thing.

### 18.1 Assertions

A proposition someone or something is putting into the record or into the model.

### 18.2 Evidence

The support or attack structure around an assertion or issue.

### 18.3 Conclusions

A higher-level synthesized position that depends on assertions, evidence,
assumptions, and context.

If these layers are blurred, the system will promote untested assertions into
conclusions and later cite those conclusions as if they were record support.

That is exactly the kind of circularity the product must prevent.

## 19. Issue Modeling And Theory Management

The system should maintain a structured issue tree, not just a list of topics.

Depending on workflow, issue nodes may include:

- litigation claims;
- defenses;
- contract interpretation disputes;
- conditions precedent;
- waiver questions;
- damages questions;
- diligence red flags;
- compliance failures;
- procedural barriers;
- evidentiary bottlenecks.

Each issue should know:

- what supports it;
- what attacks it;
- what remains unknown;
- what assumptions carry it;
- what law governs it;
- what evidence is still needed;
- how material it is;
- how strategically salient it is.

The system should also maintain multiple theories where appropriate.

For example:

- one theory may be stronger doctrinally;
- another theory may be more practical before a given judge;
- one theory may depend on a disputed factual assumption;
- another may survive even if that assumption fails.

The system should be able to preserve and compare those theories instead of
collapsing too early into one answer.

## 20. Materiality, Salience, And Practical Context

Not everything that is relevant is important, and not everything that is legally
important is strategically useful.

The system must separately model:

- relevance;
- materiality;
- legal significance;
- practical salience;
- strategic usefulness.

Examples:

- A fact may be legally relevant but strategically minor.
- A legal theory may be doctrinally elegant but useless in a forum where the
  judge has already signaled disinterest.
- A weak but practical argument may matter more than a stronger but ignored one.
- A client objective may make speed or leverage more important than exhaustive
  doctrinal coverage.

This is why the product needs a decision-context layer.

The system should be able to say:

- legally strong but strategically low-value;
- weakly provable but potentially persuasive;
- operationally important for settlement but not central to motion practice;
- worth surfacing for client risk management even if not dispositive in court.

## 21. Decision-Context Modeling

There is often critical context that does not live in the documents.

Examples:

- a judge has already indicated what kinds of arguments they care about;
- a partner wants a narrow memo, not a theory-of-the-world synthesis;
- a client wants a settlement-oriented analysis, not a maximalist merits brief;
- budget or time limits constrain how deep research should go;
- a law firm playbook disfavors certain arguments;
- the real decision-maker is commercial, not legal.

The system must be able to ingest and preserve this context.

Important rule:

Decision context should influence:

- prioritization;
- output ranking;
- clarification behavior;
- recommendation framing;
- research emphasis.

Decision context should not directly rewrite:

- what documents say;
- what the canonical record model contains;
- what assertions are grounded where;
- what the legal model says.

That separation is non-negotiable.

## 22. Uncertainty, Missingness, And Abstention

The product must be comfortable living with incomplete information.

It should explicitly model:

- uncertain authorship;
- uncertain dates;
- uncertain signatories;
- uncertain document roles;
- uncertain issue links;
- uncertain quantitative mappings;
- missing key documents;
- missing context;
- missing authorities;
- unresolved contradictions.

The correct behavior is often not "decide harder." It is:

- flag the uncertainty;
- preserve the assumption;
- ask a targeted clarification question if worth it;
- maintain multiple branches if necessary;
- abstain from overconfident conclusion where the record does not justify one.

The system must be able to say:

- insufficient record;
- disputed;
- likely but weakly supported;
- depends on missing document X;
- depends on user confirmation of context Y.

Trustworthy abstention is a required behavior, not a fallback failure mode.

## 23. Clarification Engine

The system should not ask the user random questions. It should ask only when the
question has high expected value.

The clarification engine should reason about:

- what uncertainty is active;
- what conclusions depend on it;
- whether the user is likely to know the answer;
- whether the answer will materially change routing or conclusions;
- whether another autonomous pass is less efficient than asking.

High-value clarification examples:

- "We think there may be a signed amendment that changes this clause. Do you
  know if one exists?"
- "The strategic ranking changes a lot depending on whether this judge disfavors
  standing arguments. Is there context we should account for?"
- "Damages exposure changes materially depending on whether this payment was
  actually made. Do you have confirmation?"

Clarification prompts should always say:

- what is uncertain;
- why it matters;
- what the system will do differently based on the answer.

## 24. User-Visible Reasoning And Interruptibility

Observability is not only for developers. Users need operational visibility into
the run itself.

The user should be able to see, during execution:

- what objective the system believes it is working on;
- what issue branch it is currently exploring;
- what assumptions are currently carrying the analysis;
- what evidence changed the system's view most recently;
- what documents or sections it is looking at;
- what contradictions or gaps it has found;
- what it wants to do next;
- why it chose that next step.

The user should be able to intervene by:

- stopping the run;
- redirecting the run;
- annotating a document;
- marking a source as low-trust or high-trust;
- telling the system that a line of analysis is strategically irrelevant;
- adding behind-the-scenes context;
- changing urgency or objective;
- correcting a factual assumption;
- pinning or rejecting a theory.

This is not a request for raw hidden chain-of-thought. It is a request for a
structured reasoning ledger that is safe, legible, and actionable.

## 25. User Input Must Update Different Layers Differently

One of the most important interaction design principles is that not all user
input is the same.

Different user inputs should update different layers:

- factual correction updates canonical matter memory;
- source-role correction updates source calibration and document cards;
- strategic instruction updates decision-context;
- urgency change updates attention allocation;
- presentation preference updates output behavior;
- rejection of a conclusion updates support status and issue treatment;
- repeated navigation behavior improves retrieval and ranking heuristics;
- accepted or rejected work product improves future prioritization.

If all user input is dumped into one generic memory bucket, the system will
become confused and future outputs will blur truth, preference, and strategy.

## 26. Always-On Background Maintenance Loops

The product should not only become smarter during a user query. It should
continuously improve the matter model.

Required maintenance loops include:

- repository profiling;
- duplicate and near-duplicate detection;
- document family construction;
- document-type classification;
- source-role and agenda estimation;
- actor resolution;
- speech-act extraction;
- assertion normalization;
- support and attack linking;
- contradiction mining;
- supersession and operative-status analysis;
- issue coverage analysis;
- gap detection;
- quant extraction and reconciliation;
- authority-need detection;
- user-feedback assimilation.

Why this matters:

If all of this happens only at query time, then every query is forced to pay for
basic matter understanding before it can answer the actual question. That is bad
for latency, bad for cost, and bad for cumulative quality.

## 27. Query-Time Runtime Behavior

At query time, the system should not begin with "let me read the repo."

It should begin with:

- what do I already know about this matter;
- what does the user want now;
- where is the current model weak with respect to that objective;
- which missing information is most likely to move the answer.

The runtime should:

- classify the user's task;
- inspect the current matter model first;
- identify whether the bottleneck is record, law, quant, or strategic context;
- generate a visible plan;
- retrieve targeted materials only where needed;
- preserve multiple branches where the model is unresolved;
- update durable state selectively where durable learning occurs;
- stop when additional compute has low expected value and surface the remaining
  uncertainty or missingness instead.

The runtime should optimize for moving the matter model forward, not just for
producing an answer-shaped paragraph.

## 28. Research Must Be A First-Class Layer

Research is not an optional garnish on top of document reading.

The system should know when a matter question is blocked by:

- missing law;
- unresolved procedural rule;
- venue-specific practice;
- adverse authority risk;
- lack of authority on an issue element.

The research layer should:

- maintain authorities as structured objects;
- map them to issue elements;
- search for supportive and adverse authority by default;
- identify limiting and distinguishing authority;
- integrate procedural posture and forum context;
- inform the issue model rather than float separately as citations.

Research quality should not be judged by "how many cases were found." It should
be judged by whether issue uncertainty was meaningfully reduced.

## 29. Quantitative Intelligence Must Be First-Class

Charts are a later surface, but quantitative structure is not.

The system must be able to model:

- amounts;
- dates;
- rates;
- formulas;
- balances;
- payment histories;
- invoice chains;
- damages assumptions;
- exposure scenarios;
- reconciliations;
- contradictions between numeric sources.

Many legal questions are partly numerical:

- damages;
- payment disputes;
- compliance timing;
- accrual periods;
- notice periods;
- threshold triggers;
- interest calculations;
- revenue or cost changes over time.

If the product cannot represent numbers structurally, it will mis-handle a large
class of important legal work.

## 30. Visual Work Product Comes Later But Must Be Anticipated Now

Visuals are downstream of structured state, not a substitute for it.

Useful visuals may include:

- timelines;
- issue-evidence matrices;
- damages waterfalls;
- payment-over-time charts;
- communication maps;
- version diffs;
- contradiction maps;
- obligation trackers;
- authority-to-issue maps.

These should not be generated from loose prose. They should be rendered from
structured, grounded objects.

That means the data model must be designed from the start so that future visual
surfaces are natural views over the same matter intelligence substrate.

## 31. Efficiency Is Part Of Correctness

At the scale envisioned for this product, efficiency is not merely an ops
concern. It directly affects quality and product viability.

The system should be designed around:

- cold-path ingestion that builds durable structure;
- hot-path query serving that mostly reads and selectively refreshes structure;
- dependency-aware invalidation when new docs or user corrections arrive;
- remembering productive and unproductive search paths;
- avoiding re-reading stable low-value regions;
- allocating expensive compute where expected value is highest.

Core efficiency rule:

Never spend expensive tokens to rediscover stable structure already known to the
platform.

Core quality rule:

Never choose efficiency techniques that destroy provenance, uncertainty, or the
ability to revise beliefs correctly.

## 32. Observability Requirements

Observability must exist at multiple levels.

### 32.1 Matter-Level Observability

The system should expose the state of the matter model itself:

- repository completeness;
- issue coverage;
- active contradictions;
- unsupported conclusions;
- assumption load;
- missing-document risk;
- authority gaps;
- quantitative reconciliation state;
- maturity of different issue areas.

### 32.2 Runtime Observability

The system should know, and users should be able to inspect:

- what path was taken;
- what documents were read;
- why they were selected;
- what changed;
- what branches were explored;
- where time or tokens were spent;
- what ended up being waste;
- what is queued next.

### 32.3 User-Facing Observability

The user should have a coherent live view of:

- current objective;
- current issue branch;
- current assumptions;
- current gaps;
- recent evidence changes;
- next action;
- reason for next action.

### 32.4 Learning Observability

The platform should understand where it keeps failing or being corrected:

- which source types are often misread;
- which issue types are often under-covered;
- which assumptions are often wrong;
- where user clarification most often changes the answer;
- which retrieval paths are often dead ends.

The key principle is that observability should support better reasoning, better
steering, and better future prioritization, not just system debugging.

## 33. The Product Must Improve From User Interaction

Users are not only consumers of output. They are a source of signal.

The product should improve when users:

- correct facts;
- identify missing context;
- tell the system what the judge cares about;
- indicate that a line of argument is strategically irrelevant;
- accept or reject a theory;
- point out a missing document;
- clarify a relationship between actors;
- supply a missing quantitative input;
- tell the system which output forms are useful.

However, user input must be handled carefully.

Important distinction:

- user truth corrections should affect canonical matter understanding;
- user strategy preferences should affect decision-context;
- user presentation choices should affect output mode;
- user one-off instructions should not always become durable global policy.

This distinction is crucial for keeping the matter model clean.

## 34. What Every Important Conclusion Should Be Able To Explain

Every important conclusion, recommendation, or artifact should be explainable in
roughly the following terms:

- what record material supports this;
- what source roles are involved;
- what assumptions carry it;
- what attacks or counterpoints exist;
- what issue it bears on;
- what strategic context changes its importance;
- what uncertainties remain;
- what would most likely change the conclusion.

If the system cannot provide this kind of explanation, then the conclusion is
probably not ready to be trusted.

## 35. Universal Behaviors Required In Every Matter

No matter the matter type, workflow, or user objective, the product must always:

- know what documents exist;
- know what documents are missing or likely missing;
- know who is speaking and in what role;
- distinguish advocacy, operation, procedure, and authority;
- preserve exact grounding to source spans;
- distinguish assertion from evidence from conclusion;
- distinguish record truth from inferred reality;
- distinguish legal correctness from practical value;
- surface uncertainty and assumptions;
- ask for clarification when that beats more autonomous compute;
- allow the user to see and steer the run;
- update the durable matter model when meaningful learning occurs;
- reuse durable structure aggressively;
- maintain the separation between canonical truth, strategic overlay, and
  ephemeral scratch reasoning.

## 36. Priority Order

This is the intended product priority stack.

### 36.1 Priority 0: Intelligence Substrate

These are the most important capabilities:

- repository intelligence;
- document cards;
- actor/contact store;
- source-role and agenda modeling;
- assertion and evidence graph;
- issue model;
- assumption model;
- gap model;
- user-visible reasoning ledger;
- user steering and interruptibility;
- clarification engine;
- belief revision and incremental recompute;
- matter-level and runtime observability.

If these are weak, everything else will be fragile.

### 36.2 Priority 1: Higher-Order Reasoning Layers

After the substrate is sound:

- decision-context overlays;
- proof-aware reasoning;
- adversarial reasoning;
- legal research layer;
- quantitative intelligence layer;
- better attention allocation;
- better user-feedback learning.

### 36.3 Priority 2: Presentation And Specialized Surfaces

After the structure beneath them is strong:

- richer workflow-specific modes;
- advanced presentation surfaces;
- charting;
- timelines;
- matrices;
- communication maps;
- visual analytic layers.

This ordering matters. Visual polish before structured intelligence will create
attractive but brittle product behavior.

## 37. Anti-Patterns Future Work Should Resist

Future work should explicitly resist these temptations:

- adding more prompt complexity instead of improving persistent state;
- adding more recursive search instead of better issue and repository models;
- storing extracted "facts" as free text when a structured assertion is needed;
- treating user chat history as a substitute for durable matter memory;
- optimizing for memo fluency instead of support structure;
- merging strategy, truth, and preference into one store;
- adding visual features that are not grounded in structured state;
- using repeated LLM passes to compensate for weak source-role modeling;
- ignoring missingness because the system can still produce prose;
- hiding uncertainty because it makes the answer look cleaner.

## 38. Product Success Criteria

The right high-level success criteria are not:

- answer sounds smart;
- answer is long;
- answer cites some documents;
- user can get a memo quickly.

The right success criteria are closer to:

- the system builds an accurate durable model of the matter;
- the system keeps distinct reasoning layers separate;
- the system updates correctly when new docs or user input arrive;
- the system exposes enough reasoning state for users to steer it;
- the system identifies missingness, contradictions, and weak assumptions;
- the system improves from user interaction;
- the system reuses durable structure efficiently;
- the system produces outputs that are both legally grounded and practically
  useful.

## 39. The Final Product Definition

The ideal product is a continuously maintained legal intelligence substrate for
each matter.

It should know:

- what is in the repository;
- what is missing;
- who said what;
- in what role they said it;
- why that source should be weighted the way it is;
- what is operative;
- what is superseded;
- what is disputed;
- what likely happened;
- what can be proved;
- what law applies;
- what context changes what matters;
- what assumptions are active;
- what remains uncertain;
- what the user most likely needs next.

It should let users:

- watch reasoning as it happens;
- intervene while the system is working;
- add context the documents do not contain;
- correct wrong assumptions;
- redirect effort;
- get reusable structured work product, not only prose.

It should improve:

- as documents arrive;
- as users correct it;
- as it learns which paths were useful;
- as it notices what the matter model is still missing.

And it should treat every text answer, chart, timeline, research packet, and
memo as a downstream view over that evolving matter model.

That is the product north star.
