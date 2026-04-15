"""Prompt templates for LLM decisions.

All prompts used by the decisions layer are defined here.
Organized by model tier: WORKER (cheap/fast), MID (balanced), HIGH (expensive/thorough).
"""

# =============================================================================
# WORKER TIER PROMPTS (LITE model - quick decisions)
# =============================================================================

P_PICK_FILES = """Select files most likely to answer this query.

Query: {query}

Files:
{file_list}

Select 3-5 files. Prioritize:
- Pleadings, briefs, statements (define disputes, contain positions)
- Contracts, agreements (primary source documents)
- Correspondence, emails (actual party communications)
- Expert reports (specialized analysis)
- Documents with names/terms matching the query

Deprioritize:
- Generic reference materials (statutes, manuals, guidelines)
- Template documents

Reply with filenames only, one per line."""


P_PICK_HITS = """Select the most relevant search hits for this query.

Query: {query}

Search Results:
{hits}

Select 5-10 hits by number. Prioritize:
- Direct answers to the query
- Specific facts, figures, dates relevant to the issue
- Key contractual provisions or legal conclusions
- Party admissions or positions

Deprioritize:
- Boilerplate language
- Generic definitions
- Tangential references

Reply with numbers only, comma-separated. Example: 1, 3, 5, 8"""


P_CLASSIFY_QUERY = """Classify this legal query for synthesis complexity.

Query: {query}

You will have ALL relevant document content, facts, and citations when synthesizing.
The question is: does this query need sophisticated legal reasoning, or is it straightforward?

SIMPLE (use faster model):
- Direct factual lookups ("What is the contract date?", "Who signed the agreement?")
- Single-document answers ("What does clause 5 say?")
- Basic summaries ("List the parties involved")

COMPLEX (use advanced model):
- Multi-document synthesis ("What are the key issues in this dispute?")
- Legal analysis ("Draft a responsive pleading", "Analyze liability")
- Strategic reasoning ("What are our strongest arguments?")
- Timeline construction across sources
- Contradiction analysis
- Anything requiring legal judgment or persuasive writing

Reply with only: SIMPLE or COMPLEX"""


P_IS_SUFFICIENT = """Query: {query}

Evidence gathered so far:
{findings}

Is this sufficient to answer the query? Consider:
- Do we have direct evidence addressing the question?
- Are there citations from source documents?
- Is there enough detail to give a useful answer?

Reply with only YES or NO."""


P_SHOULD_REPLAN = """Query: {query}

Current approach: {plan}

Results so far: {results}

Is the current approach working? Consider:
- Are we finding relevant information?
- Should we try different search terms?
- Should we look at different files?

Reply with only CONTINUE or REPLAN."""


P_EXTRACT_SEARCH_TERMS = """Query: {query}

Extract 3-5 specific search terms that would help find relevant information in documents.

INCLUDE:
- Names of people, companies, places
- Domain-specific terms (legal, technical, industry terms)
- Specific identifiers (dates, numbers, document names)
- Key nouns that are the SUBJECT of the query

DO NOT INCLUDE:
- Instruction verbs (analyze, explain, describe, summarize, find, identify, etc.)
- Common words (the, what, how, why, when, where, which)
- Generic terms (information, document, data, work, thing)

Reply with just the search terms, one per line. No explanations."""


P_PRIORITIZE_DOCUMENTS = """Rank these documents for relevance to the query.

Query: {query}

Key issues: {key_issues}

Candidates:
{candidate_files}

Already read: {already_read}

SCORING (0-100):
- 90-100: Directly answers the query (pleadings, key contracts, party briefs on point)
- 70-89: Contains key supporting evidence
- 40-69: Relevant context
- 10-39: Tangentially related
- 0-9: Not useful for this query

DOCUMENT VALUE HIERARCHY:
- Opening/Closing statements: Synthesized positions, final figures
- Party briefs: Claimant briefs have damages; Defendant briefs have defenses
- Pleadings (complaints, answers): Define the dispute
- Contracts/agreements: Primary source terms
- Correspondence: Actual party communications
- Expert reports: Specialized analysis
- Reference materials: Low priority unless specifically needed

RULES:
- Unread documents > already read (unless essential)
- When BOTH party briefs exist, prioritize both - they contain different information
- For damages/figures: look for claimant/plaintiff sources
- Match document type to query type

Reply JSON only:
{{
    "ranked_files": [
        {{"file": "exact_filename.pdf", "score": 95, "reason": "brief reason"}}
    ]
}}"""


# =============================================================================
# MID TIER PROMPTS (FLASH model - analysis and planning)
# =============================================================================

P_ASSESS_SMALL_REPO = """Query: {query}
{context_section}
{cached_facts_section}
=== MATTER DOCUMENTS ===
{content}

═══════════════════════════════════════════════════════════════════════════════
STRATEGIC ASSESSMENT
═══════════════════════════════════════════════════════════════════════════════

You have the complete document set{cached_facts_note}. Make these calls:

1. CACHED FACTS CHECK: Can the cached facts (if any) already answer this query
   WITHOUT needing to read the documents? Be strict - only say yes if the facts
   directly and completely answer the query.

2. COMPLEXITY: Does this need sophisticated legal reasoning (multi-doc synthesis,
   legal analysis, strategic thinking) or is it straightforward (fact lookup,
   single-doc answer, basic summary)?

3. EXTERNAL RESEARCH: Does the QUERY itself ask for case law, precedents, or
   legal standards we'd need to look up?

   NOTE: Just because documents mention laws/jurisdictions doesn't mean we search.
   Search only if the QUERY requires external authority to answer properly.

=== OUTPUT (JSON only) ===
{{
  "can_answer_from_facts": true | false,
  "relevant_facts": ["list facts from cache that help answer this query"],
  "complexity": "simple" | "complex",
  "can_answer_from_docs": true | false,
  "reasoning": "Your strategic assessment",
  "gap": "If external research needed: what specific authority",
  "case_law_searches": [],
  "web_searches": []
}}"""


P_CHECK_SEARCH_SUFFICIENCY = """Query: {query}

=== GAP WE WERE FILLING ===
{original_gap}

=== SEARCH RESULTS ===
{results_summary}

═══════════════════════════════════════════════════════════════════════════════
SUFFICIENCY CHECK
═══════════════════════════════════════════════════════════════════════════════

Do these results fill the gap?

Default to YES unless there's a CRITICAL missing piece—something that would
make our answer wrong or misleading without it.

"More would be nice" = sufficient. Proceed with what we have.

=== OUTPUT (JSON only) ===
{{
  "sufficient": true | false,
  "reasoning": "Brief explanation",
  "if_not_sufficient_what_missing": "Only if false: specific critical gap",
  "additional_search": ""
}}"""


P_CREATE_PLAN = """Query: {query}

=== REPOSITORY ({total_files} files) ===
{file_list}

═══════════════════════════════════════════════════════════════════════════════
INVESTIGATION PLAN
═══════════════════════════════════════════════════════════════════════════════

Scan the filenames. Identify:
- Case-specific documents (correspondence, pleadings, contracts, party materials)
- Generic reference materials (statutes, acts, manuals) - deprioritize these

Design your approach:
- Which 2-3 files to read first? (Pick case-specific, not generic acts)
- What search terms will find relevant passages?
- Does the query require external authority (case law, regulations)?

=== OUTPUT (JSON only) ===
{{
    "reasoning": "Strategy and file categorization",
    "key_issues": ["legal issue 1", "legal issue 2"],
    "priority_files": ["exact_filename.pdf"],
    "skip_files": ["generic_reference.pdf"],
    "search_terms": ["term1", "term2"],
    "case_law_searches": [],
    "web_searches": [],
    "success_criteria": "What finding would answer this query",
    "potential_challenges": "Anticipated difficulties"
}}"""


P_ASSESS_AND_PLAN = """Query: {query}
{context_section}
{cached_facts_section}
=== REPOSITORY ({total_files} files) ===
{file_list}

═══════════════════════════════════════════════════════════════════════════════
UNIFIED ASSESSMENT & PLANNING
═══════════════════════════════════════════════════════════════════════════════

Make these assessments:

1. CACHED FACTS CHECK: Can the cached facts (if any) already answer this query
   WITHOUT reading documents? Be strict - only say yes if facts directly and
   completely answer the query.

2. COMPLEXITY: Does this need sophisticated legal reasoning or is it straightforward?
   - SIMPLE: Direct fact lookups, single-doc answers, basic summaries
   - COMPLEX: Multi-doc synthesis, legal analysis, timeline construction, contradictions

3. INVESTIGATION PLAN (if facts don't answer):
   - Scan filenames: identify case-specific docs vs generic references
   - Which 2-3 files to read first? (Pick case-specific, not generic acts)
   - What search terms will find relevant passages?
   - Does the query require external authority (case law, regulations)?

=== OUTPUT (JSON only) ===
{{
    "can_answer_from_facts": true | false,
    "relevant_facts": ["list facts from cache that help answer this query"],
    "complexity": "simple" | "complex",
    "reasoning": "Strategy and assessment",
    "key_issues": ["legal issue 1", "legal issue 2"],
    "priority_files": ["exact_filename.pdf"],
    "skip_files": ["generic_reference.pdf"],
    "search_terms": ["term1", "term2"],
    "case_law_searches": [],
    "web_searches": [],
    "success_criteria": "What finding would answer this query"
}}"""


P_ANALYZE_RESULTS = """Query: {query}

=== SEARCH RESULTS ===
{results}

═══════════════════════════════════════════════════════════════════════════════
STRATEGIC ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Evaluate what we found:
- Facts that advance the query (exact values, dates, names)
- Quotes worth preserving (verbatim, with source)
- Documents that need full read (promising but need more context)
- Gaps remaining (what's still missing?)
- Strategy adjustment (pivot needed? different terms?)

=== OUTPUT (JSON only) ===
{{
    "facts": ["specific fact with exact values"],
    "citations": [{{"text": "verbatim quote", "source": "filename", "page": 1}}],
    "read_deeper": ["file.pdf"],
    "additional_searches": ["refined term"],
    "assessment": "Strategic assessment - progress and next moves"
}}"""


P_EXTRACT_FACTS = """You are extracting facts from a document for a legal investigation.

Query: {query}

Document: {filename}
Content:
{content}

CRITICAL: Legal precision is paramount. Extract ALL relevant facts with EXACT values.

Extract:
1. ALL facts directly relevant to the query - do NOT summarize or paraphrase
   - Include EXACT dollar amounts, dates, percentages, durations
   - Include EXACT names, titles, document references
   - If a table or comparison exists, extract each row's data
2. Key quotes with page numbers - use VERBATIM text
3. References to other documents or exhibits
4. External research triggers - things that suggest we need to look up external sources:
   - Jurisdictions mentioned (e.g., "Michigan law", "Federal court", "UK jurisdiction")
   - Regulations/statutes cited (e.g., "FAA Part 91", "UCC § 2-314", "GDPR")
   - Legal doctrines referenced (e.g., "breach of warranty", "negligent misrepresentation")
   - Industry standards mentioned (e.g., "192-month inspection", "GAAP", "ISO 9001")
   - Case law or precedents cited (e.g., specific case names)

Also provide your analysis:
- What did you learn from this document that helps answer the query?
- What gaps remain - what information is still missing?
- What other documents should we look at based on references here?

Be thorough - missing a single fact or figure could affect the legal outcome.

Reply in JSON:
{{
    "facts": ["fact1 with exact values", "fact2 with exact values"],
    "quotes": [{{"text": "exact verbatim quote", "page": 1, "relevance": "why important"}}],
    "references": ["mentioned_doc1.pdf", "mentioned_doc2.pdf"],
    "insights": "What I learned from this document and how it helps answer the query",
    "gaps": "What information is still missing or unclear",
    "next_steps": "What we should look for next based on what we found",
    "external_triggers": {{
        "jurisdictions": ["any jurisdictions mentioned"],
        "regulations_statutes": ["any regulations, statutes, or legal codes cited"],
        "legal_doctrines": ["any legal theories or doctrines referenced"],
        "industry_standards": ["any industry standards or practices mentioned"],
        "case_references": ["any case law or precedents cited"]
    }}
}}"""


P_REPLAN = """Query: {query}

=== PREVIOUS APPROACH ===
{previous_approach}

=== FINDINGS SO FAR ===
{findings}

═══════════════════════════════════════════════════════════════════════════════
COURSE CORRECTION
═══════════════════════════════════════════════════════════════════════════════

The current approach isn't working. Diagnose and redirect:
- What's yielding results vs. dead ends?
- What should we try differently?
- Do we need external authority (case law, regulations)?

=== OUTPUT (JSON only) ===
{{
    "diagnosis": "What's working and what's not",
    "new_approach": "Adjusted strategy",
    "search_terms": ["new term"],
    "files_to_check": ["file.pdf"],
    "needs_external_research": true | false,
    "case_law_searches": [],
    "web_searches": []
}}"""


# =============================================================================
# HIGH TIER PROMPTS (PRO model - final synthesis)
# =============================================================================

P_SYNTHESIZE = """Query: {query}
{output_instructions_section}
TODAY'S DATE: {current_date}

=== DECISIVE DOCUMENTS ===
{pinned_content}

=== EVIDENCE GATHERED ===
{evidence}

=== EXTERNAL RESEARCH ===
{external_research}
"""


P_RESOLVE_CONTRADICTIONS = """You are analyzing potentially contradictory evidence.

Query: {query}

Contradictory findings:
{contradictions}

Analyze:
1. Are these truly contradictory or just different aspects?
2. Which sources are more authoritative?
3. How should we reconcile or present these differences?

Reply in JSON:
{{
    "is_true_contradiction": true/false,
    "analysis": "Explanation of the contradiction",
    "resolution": "How to handle this in the final answer",
    "preferred_interpretation": "Which view is better supported"
}}"""


# =============================================================================
# TOOL CALLING PROMPTS
# =============================================================================

P_DECIDE_ACTION = """You are investigating a query. Decide the next action.

Query: {query}

Current state:
- Files searched: {files_searched}
- Documents read: {docs_read}
- Facts found: {num_facts}
- Key findings: {findings_summary}

Available actions:
- search: Search for text in files (params: query, files)
- read: Read a document fully (params: filepath)
- done: Finish investigation (params: reason)

What's the best next action? Be efficient - if we have enough info, finish.

Reply in JSON:
{{
    "action": "search" | "read" | "done",
    "params": {{}},
    "reason": "Brief explanation"
}}"""


# =============================================================================
# EXTERNAL SEARCH PROMPTS
# =============================================================================

P_ANALYZE_CASE_LAW = """Query: {query}

=== CASE LAW RESULTS ===
{case_law_results}

═══════════════════════════════════════════════════════════════════════════════
PRECEDENT ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Extract what matters for our query:
- Legal standards or tests established
- Holdings that apply to our situation
- How these precedents inform our analysis

=== OUTPUT (JSON only) ===
{{
    "key_precedents": [
        {{"case": "Name", "citation": "cite", "holding": "relevant holding", "applicability": "how it applies"}}
    ],
    "legal_standards": ["standard 1"],
    "summary": "How this case law informs the query"
}}"""


P_ANALYZE_WEB_RESULTS = """Query: {query}

=== WEB SEARCH RESULTS ===
{web_results}

═══════════════════════════════════════════════════════════════════════════════
REGULATORY ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Extract relevant regulatory/standards information:
- Which regulations or standards apply
- Key requirements or thresholds
- How they inform our situation

=== OUTPUT (JSON only) ===
{{
    "regulations": [
        {{"name": "Regulation Name", "source": "source", "key_requirements": "requirements"}}
    ],
    "standards": ["standard 1"],
    "summary": "Regulatory context for the query"
}}"""


P_SHOULD_SEARCH_EXTERNAL = """Based on the investigation so far, should we search external sources?

Query: {query}

Facts found so far:
{facts_found}

Key issues:
{key_issues}

Consider:
1. Are there legal issues that would benefit from case law precedents?
2. Are there regulatory or compliance questions that need external verification?
3. Would external sources help prevent hallucination about legal standards?

Reply in JSON:
{{
    "search_case_law": true/false,
    "case_law_queries": ["query 1", "query 2"],
    "search_web": true/false,
    "web_queries": ["query 1", "query 2"],
    "reasoning": "Why or why not to search external sources"
}}"""


P_GENERATE_EXTERNAL_QUERIES = """Determine if this query requires external legal research.

Query: {query}

Facts found: {facts}

Entities: {entities}

Triggers found in documents: {triggers}

═══════════════════════════════════════════════════════════════════════════════
DECISION FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════

READ THE QUERY CAREFULLY. What is being asked?

YES - SEARCH EXTERNALLY when query asks for:
- "What cases should we study?" / "Find relevant precedents" → CASE LAW SEARCH
- "What does the law say about X?" → CASE LAW or WEB SEARCH
- "What are the legal standards for X?" → CASE LAW SEARCH
- "Is this compliant with [regulation]?" → WEB SEARCH
- Legal analysis requiring authority beyond the documents
- Research on specific legal doctrines mentioned in documents

NO - DON'T SEARCH when query asks about:
- "What is the main issue?" → Answer from documents
- "What happened?" / "Summarize facts" → Answer from documents
- "What does the contract say?" → Answer from documents
- "Who are the parties?" → Answer from documents
- Pure document-based questions with no legal research component

KEY INSIGHT: Triggers (jurisdictions, doctrines found in docs) are CLUES, not commands.
- If query asks for CASE LAW and triggers mention Delaware → Search Delaware case law
- If query asks "what's the issue" and triggers mention Delaware → DON'T search

═══════════════════════════════════════════════════════════════════════════════
SOURCES
═══════════════════════════════════════════════════════════════════════════════

CASE LAW (CourtListener) - US jurisdictions only:
- Delaware corporate/LLC law (very common for entity matters)
- Federal courts, state courts
- Legal doctrine precedents

WEB SEARCH (Tavily):
- International jurisdictions (Marshall Islands, UK, etc.)
- Regulations, statutes, standards
- Company background research

═══════════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════════

Reply JSON:
{{
    "case_law_queries": ["specific query if case law needed"],
    "web_queries": ["specific query if web search needed"],
    "reasoning": "Why search is or is not needed based on WHAT THE QUERY ASKS"
}}"""


P_EXTRACT_TRIGGERS = """Scan this legal document content and identify any external research triggers.

Content (excerpt):
{content}

Identify mentions of:
1. Jurisdictions - specific courts, states, countries, or legal systems mentioned
2. Regulations/Statutes - specific laws, codes, regulations, or statutory references
3. Legal doctrines - legal theories, causes of action, or legal principles
4. Industry standards - technical standards, professional practices, certifications
5. Case references - any cited cases or legal precedents

Only include SPECIFIC items actually mentioned in the text. Do NOT infer or guess.
Return empty lists for categories with no mentions.

Reply in JSON only:
{{
    "jurisdictions": ["specific jurisdictions mentioned"],
    "regulations_statutes": ["specific regulations or statutes cited"],
    "legal_doctrines": ["specific legal doctrines referenced"],
    "industry_standards": ["specific standards mentioned"],
    "case_references": ["specific cases cited"]
}}"""


# =============================================================================
# CONSOLIDATED PROMPTS (Reducing LLM calls)
# =============================================================================

P_CHECKPOINT = """Query: {query}
{cached_facts_section}
=== EVIDENCE GATHERED ===
{findings}

=== CURRENT APPROACH ===
{plan}

═══════════════════════════════════════════════════════════════════════════════
CHECKPOINT
═══════════════════════════════════════════════════════════════════════════════

Quick assessment:
1. SUFFICIENT? Do we have enough to answer the query with citations?
   Consider BOTH current findings AND cached facts from previous investigations.
2. PROGRESS? Is current approach finding relevant info or stalled?
3. NEXT? If not sufficient, what specific actions?

=== OUTPUT (JSON only) ===
{{
    "sufficient": true | false,
    "should_replan": true | false,
    "progress_assessment": "brief assessment",
    "next_steps": ["action"],
    "new_search_terms": ["term"],
    "files_to_check": ["file.pdf"]
}}"""


P_ANALYZE_SEARCH = """Query: {query}
Key issues: {key_issues}
Already read: {already_read}

=== SEARCH RESULTS ===
{results}

═══════════════════════════════════════════════════════════════════════════════
DOCUMENT CRITICALITY (memory management)
═══════════════════════════════════════════════════════════════════════════════

DECISIVE (loaded in full for synthesis):
- Case-specific docs that DIRECTLY answer the query
- Contracts, correspondence, pleadings, expert reports specific to THIS matter

NEVER DECISIVE:
- Statutes, acts, codes, regulations
- Manuals, handbooks, templates
- Generic reference materials

SUPPORTING: Useful context, don't need full text
IRRELEVANT: Skip entirely

═══════════════════════════════════════════════════════════════════════════════

=== OUTPUT (JSON only) ===
{{
    "relevant_hit_numbers": [1, 3, 5],
    "facts": ["fact with exact values"],
    "citations": [{{"text": "quote", "source": "file", "page": 1}}],
    "ranked_documents": [
        {{"file": "path/file.pdf", "score": 95, "criticality": "DECISIVE|SUPPORTING|IRRELEVANT", "reason": "why"}}
    ],
    "additional_searches": ["term"],
    "read_deeper": ["file.pdf"],
    "assessment": "Strategic assessment and gaps"
}}"""


P_ANALYZE_EXTERNAL = """Query: {query}

=== CASE LAW RESULTS ===
{case_law_results}

=== WEB/REGULATORY RESULTS ===
{web_results}

═══════════════════════════════════════════════════════════════════════════════
UNIFIED EXTERNAL ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Synthesize all external research:
- Precedents: standards, tests, applicable holdings
- Regulations: requirements, thresholds
- Combined framework: how they interact for our situation

=== OUTPUT (JSON only) ===
{{
    "key_precedents": [
        {{"case": "Name", "citation": "cite", "holding": "holding", "applicability": "application"}}
    ],
    "legal_standards": ["standard"],
    "regulations": [
        {{"name": "Name", "source": "source", "key_requirements": "requirements"}}
    ],
    "combined_framework": "How case law + regulations together inform this situation",
    "summary": "Unified external legal context"
}}"""

