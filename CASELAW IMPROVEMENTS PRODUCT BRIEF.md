# AR Video Review Feedback — April 16, 2026
**From:** Christian Brown (Product)
**To:** Arpit / Devansh / Sudarsh
**Date:** April 16, 2026
**Reference:** Product Brief [`12-agentic-retrieval.md`](./12-agentic-retrieval.md) (v5.1) | Engineering Feedback [`12a-agentic-retrieval-engineering-feedback.md`](./12a-agentic-retrieval-engineering-feedback.md) (v1.9) | Apr 15 Video Review (integrated into #12 v5.1)

---

## Summary

Ran the same three-query pattern again today on the new AR branch, side-by-side against yesterday's branch (both URLs `agentic-retrieval.iqidis.ai`, matter `CB Test`). Three takeaways:

1. **Case law search is materially better today.** The Apr 15 Mode A (invoked-but-can't-find → silent Tavily fallback) and Mode B (Coordinator not invoking search at all) findings both look substantially improved. On all three queries the Coordinator fires case law search and returns real CourtListener results. Mode B is no longer surfacing in this test set. The Mode A Tavily fallback appears gone on these runs.
2. **Inline citation injection is firing.** Superscript markers now appear next to case names in the memo body on V1 and V2. The Apr 15 top priority ("research finds the right cases but memo doesn't cite them") is largely resolved as an injection architecture problem.
3. **The remaining failure has shifted.** Citations are being *injected*, but **the case name bound to each citation number is frequently wrong.** On V2 (citation-number-only query), 3 of 5 rows in the Citation Validation Table have a wrong case name bound to the right citation — and all 5 are marked "Valid". This is a citation↔case-name binding bug, distinct from the earlier injection drop-out and Tavily-fallback failures.

---

## Test Setup

Three queries run in the `CB Test` matter. Left panel in each video = yesterday's branch run (Apr 15). Right panel = today's branch run (Apr 16).

| # | Query | Shape |
|---|-------|-------|
| V1 | "Can you find and validate the Texas state cases? Trevino v. State, Formosa Plastics Corp. USA v. Presidio Engineers & Contractors, Inc., Kroger Co. v. Persley, City of Keller v. Wilson, In re Halliburton Co." | Case names only, no citation numbers |
| V2 | "Please validate these citations 991 S.W.2d 849 (Tex. Crim. App. 1999), 960 S.W.2d 41 (Tex. 1998), 261 S.W.3d 316 (Tex. App.—Houston [1st Dist.] 2008, no pet.), 168 S.W.3d 802 (Tex. 2005), 80 S.W.3d 566 (Tex. 2002)" | Citation numbers only, no case names |
| V3 | "Extract all case citations from this matter's documents and validate the citations." | Matter-doc extraction + validation |

V1 and V2 query the same underlying five cases (Trevino, Formosa, Kroger, Keller, Halliburton) via different handles, which is useful — it exposes whether the system treats the two forms as equivalent.

---

## What's Better on the New Branch

1. ✅ **Case law search is invoked reliably** across all three queries (Mode B fix holding).
2. ✅ **Case law search returns real CourtListener cases** populated into the Sources & Citations panel on every run (Mode A Tavily-fallback failure not observed in this test set).
3. ✅ **Inline citation markers are now injected** — superscript numbers appear next to case names in the memo body on V1 and V2.
4. ✅ **V3 memo structure is meaningfully better** — four-section format (I. Validated Citations / II. Citations With Discrepancies / III. Unvalidated Citations (Not In Evidence) / IV. Unused Evidence) replaces yesterday's flat prose memo. This is a real UX step up.
5. ✅ **V3 "Evidence Match" field** — surfaces the original case name from the matter's evidentiary documents alongside each cited name. Useful for catching party-name variants, typos, and near-miss mis-citations.
6. ✅ **V3 genuine discrepancy catch** — flagged *Senior Care Living VI, LLC v. Preston Hollow Capital, LLC* as brief citing a 2024 published opinion but referencing an unpublished docket number / earlier date. That's exactly the kind of substantive catch that makes validation worth running.
7. ✅ **V3 memo date is correct (April 16, 2026)** — the Apr 15 hallucinated-date finding is fixed on this path. (See below — V1 and V2 still need spot-check.)

---

## Primary New Issue — Citation ↔ Case-Name Binding

Confirmed on all three videos. Cleanest evidence is V2.

### V2 — Citation Validation Table

Five citation numbers queried. The right-panel Citation Validation Table returns:

| Citation queried | New branch maps to | Correct case (per V1 ground truth) | Status returned |
|------------------|-------------------|--------------------------------|-----------------|
| 991 S.W.2d 849 (Tex. Crim. App. 1999) | Trevino v. State | Trevino v. State | Valid ✅ |
| 960 S.W.2d 41 (Tex. 1998) | Formosa Plastics Corp. USA v. Presidio Eng'rs & Contractors, Inc. | Formosa Plastics | Valid ✅ |
| 261 S.W.3d 316 (Tex. App.—Houston [1st Dist.] 2008, no pet.) | **Glattly v. AirSera, LLC** | Kroger Co. v. Persley | Valid ❌ |
| 168 S.W.3d 802 (Tex. 2005) | **BMC Software Belgium, N.V. v. Marchand** | City of Keller v. Wilson | Valid ❌ |
| 80 S.W.3d 566 (Tex. 2002) | **Reata Construction Corp. v. City of Denton** | In re Halliburton Co. | Valid ❌ |

**Three of five rows bind the wrong case name to the right citation. All five rows show "Valid".**

The key diagnostic: the **Sources & Citations panel on the same V2 run includes `Kroger Co. v. Persley — 261 S.W.3d 316 — Texas Court of Appeals, 1st District (Houston)`** as a retrieved citation. The case law search retrieved the correct case for 261 S.W.3d 316 — but the Citation Validation Table synthesis picked a different retrieved candidate (Glattly v. AirSera) for that row. The data is in the retrieval output; the binding step isn't using it.

Yesterday's branch (left panel) showed the same misbinding pattern in its memo body ("3. Glattly v. Air-Cruisers Co, 261 S.W.3d 316 / 4. Michiana Easy Livin' Country … 168 S.W.3d 802 / 5. BMC Software Belgium … 80 S.W.3d 566"). Today's branch picked *different* wrong cases for 80 S.W.3d 566 (Reata Construction vs. yesterday's BMC Software). **The wrong answers are non-deterministic across runs, which suggests candidate-ranking variance rather than a fixed wrong mapping.**

### V1 — Case Names Query, Trevino Multiples

V1 queries by case name. Sources panel shows 22 citations from 22 sources. For "Trevino v. State" the panel lists five different Trevino individuals across five different Texas appellate districts:

- Bobby Trevino v. the State of Texas (Texas Court of Appeals, 12th District / Tyler)
- Melissa Trevino v. the State of Texas (13th District)
- Abel Trevino v. the State of Texas (13th District)
- Jorge Trevino Cardenas v. the State of Texas (1st District / Houston)
- Joshua Lee Trevino v. the State of Texas (9th District / Beaumont)

None of these is Trevino v. State, 991 S.W.2d 849 (Tex. Crim. App. 1999) — the foundational Rule 404(b) extraneous-offense evidence precedent from the Court of Criminal Appeals. The memo body's holding text describes the correct case (Rule 404(b), extraneous offenses, Rule 403 balancing), but the inline citation marker next to "Trevino v. State" links to one of these five wrong Trevino appellate cases. **Click-through from the inline citation takes the reader to the wrong opinion.**

### V3 — Harder to See, Worth Spot-Checking

V3's Evidence Match pattern largely masks this class of bug because both sides of the match are drawn from the same matter's documents. Worth verifying whether the Evidence Match row for each validated citation is actually the same case as the Brief Location row, or whether the match is binding by citation number with some tolerance.

---

## Diagnostic Hypothesis

The pattern is consistent with a **citation lookup that returns multiple candidates and selects by something other than canonical citation match**.

Two contributing factors appear likely:

**(1) Citation-number queries are hitting Search API, not Citation Lookup API.** V2's Sources panel includes cases like `State ex rel. S.W.`, `In re D.L.S.W.`, `Victor Lissiak Jr. v. S.W. Loan OG L.P.`, `A.N.S.W.E.R. v. Norton` — none of which are at the queried reporter positions; they are cases where "S.W." appears as a substring of a party name. That's a keyword-search artifact. When a user pastes `80 S.W.3d 566`, the pipeline should be calling Citation Lookup API (which returns the canonical case at that exact volume/reporter/page) — not the Search API, which tokenizes "S.W." as a keyword and surfaces noise.

**(2) Case-name queries aren't disambiguating by citation number.** V1's five Trevino results are all real Texas appellate decisions, but none is at 991 S.W.2d 849. A "Trevino v. State" search is returning the top-N matches by name without cross-checking against the citation number the user provided.

Either way, the downstream binder in the Validation Table / Synthesizer is picking from a candidate pool that's already polluted with the wrong cases, and its selection logic doesn't appear to be "which retrieved case has the exact citation number match."

---

## Engineering Questions

1. **Citation Validation Table — where does "Case Name" come from?** Is it populated from the Sources panel's retrieved metadata (and if so, which ranked candidate), or is it resolved independently? V2 sources had `Kroger Co. v. Persley, 261 S.W.3d 316` but the table showed `Glattly v. AirSera, LLC` for the same citation — so at minimum the binding isn't "first source panel entry at that citation."

2. **Citation-number vs. case-name query paths.** When the query is a reporter citation (e.g., `991 S.W.2d 849`), is the pipeline calling Citation Lookup API first, or falling to Search API? V2's Sources panel composition strongly suggests the Search API path is being hit, which is both slower and noisier than Lookup for exact-citation queries.

3. **"Valid" status semantics.** V2's Citation Validation Table marks all 5 rows "Valid" when 3 have wrong case-name bindings. What does "Valid" mean today — that the citation string parses? That any case exists at that reporter location? That the chosen case name matches the canonical case at that citation? If the last, the logic is wrong. If either of the former, the status label is actively misleading and needs to change.

4. **Inline citation target resolution.** When memo body reads `Trevino v. State, 991 S.W.2d 849 (Tex. Crim. App. 1999) [1]`, what determines where `[1]` points? Is it the nearest case-name match in Sources (which picks one of the wrong Trevinos), or the nearest citation-number match (which should pick the right one if it's retrieved)? Logging the resolution logic here would help.

5. **Trevino-multiples dedup / disambiguation.** V1 sources returned 5 different Trevino individuals' appellate cases. Does the case law search do any citation-number disambiguation when the user provides both a case name AND a citation number? If not, that's the first thing to wire in — it's the single highest-leverage fix for the V1 case.

6. **Candidate ranking determinism.** On V2, yesterday's branch bound `80 S.W.3d 566` → BMC Software and today bound it → Reata Construction. Different wrong answers across runs for the same citation suggests ranking non-determinism. What's the ranking signal (Gemini score, search relevance, insertion order)? Can we pin it?

7. **Structured logging ask for the next push.** For each citation that enters the memo body or Validation Table, log: (a) the full candidate list returned by search, (b) the ranking signal used, (c) the selected case, (d) why it was chosen over higher-ranked candidates. Without that visibility we can't separate "wrong case retrieved" (search problem) from "right case retrieved, wrong case bound" (synthesizer problem) — and those need different fixes.

---

## Smaller But Worth Fixing

---

## Proposed Priority Reshuffle (vs. v5.1 Brief)

Given today's test, the Apr 15 production-blocker stack should be updated:

| Brief v5.1 item | Status today | Action |
|---|---|---|
| 🔴 Post-generation citation injection drop-outs (Apr 15 #1) | Largely resolved — inline citations present on all runs | Downgrade to 🟡; confirm across a broader query set |
| 🔴 Case law search Mode A (Tavily fallback) | Not observed today | Downgrade to 🟡; rerun v2 (31-case MSJ validation) to confirm |
| 🔴 Case law search Mode B (Coordinator skip) | Not observed today | Downgrade to 🟡 |
| 🔴 Case law pipeline rework (Perplexity → CourtListener → full opinion) | Partially landing — need confirmation of Perplexity pre-filter step | Keep 🔴 pending verification |
| 🔴 SSE stream timeout | No change — still a blocker on long investigations | Keep 🔴 |
| — | **NEW** | 🔴 **Citation ↔ case-name binding.** Retrieved cases not correctly matched to citation numbers in memo body / Validation Table. Highest-visibility quality issue in today's output. |
| — | **NEW** | 🔴 **Citation Lookup API on citation-number queries.** V2 Sources panel suggests Search API path is being hit for pure-citation queries, which produces noise. |
| — | **NEW** | 🟡 **"Valid" status on mis-bound rows.** Validation Table marks all "Valid" when 3 of 5 have wrong bindings. Status label is misleading until the binding is fixed. |

---

*v1.0 — April 16, 2026. First writeup of today's three-video side-by-side test. Documents citation↔case-name binding as the new top quality issue, with Apr 15 injection drop-out and Tavily-fallback issues largely resolved on this test set. Hypothesizes Citation Lookup vs. Search API path as a contributing cause. Proposes structured logging on binding step as the next-sprint ask.*