# Irys Knowledge Architecture: Why Assertions Matter

## Summary for Non-Technical Stakeholders

**Author:** Technical Discussion Summary
**Date:** January 2026
**Status:** Design Discussion (Not Yet Implemented)

---

## The Problem We're Solving

### Current Limitation: Irys Has No Memory

Right now, every time someone asks Irys a question about a legal matter, it starts completely fresh. It reads all the documents again, extracts information again, and builds an answer from scratch.

**Example:**
- Monday: User asks "What's the contract value?" → Irys reads 50 documents, finds the answer
- Tuesday: User asks "What's the contract value?" → Irys reads the same 50 documents again

This is like having a research assistant who forgets everything overnight. It works, but it's:
- **Slow** - Repeating work that's already been done
- **Expensive** - Paying for AI processing we've already paid for
- **Missing opportunity** - Knowledge from one question could help answer the next

### The Obvious Solution (That Doesn't Work)

The obvious fix: Store what we learn. Build a "fact database" for each matter.

**But this is dangerous in legal contexts.**

Why? Because in legal documents, "facts" aren't simple truths. They're claims made by parties who have interests and agendas.

---

## Why Legal "Facts" Are Different

### Problem 1: Same Question, Different Answers

In a typical database, a fact is a fact. "The sky is blue" is either true or false.

In legal documents:
- **Plaintiff says:** "The defendant breached the contract on June 1"
- **Defendant says:** "We fully performed all our obligations"

Both of these appear in the case documents. Both parties state them confidently. If we just store "facts," which one is the fact?

**Reality:** In legal matters, we're not storing facts. We're storing *claims made by specific parties*.

### Problem 2: Facts Change Over Time

- **January:** Original contract says "$1,000,000"
- **March:** Amendment changes it to "$1,500,000"
- **June:** Second amendment changes it to "$5,000,000"

If we stored "Contract value is $1M" as a fact in January, that "fact" is now wrong. But it wasn't wrong in January - it was true then.

**Reality:** Legal facts have time dimensions. Something true in January may not be true in June.

### Problem 3: Context Gets Lost

Document says: *"The warranty is void **if payment is not received within 30 days**"*

A naive system might extract: *"The warranty is void"*

Now we've turned a conditional statement into an absolute one. Someone asks "Is the warranty valid?" and we confidently say "No" - completely missing the crucial condition.

**Reality:** Extracting information without its context can change its meaning entirely.

---

## Our Solution: Assertions Instead of Facts

Instead of storing "facts," we store **assertions** - which preserve crucial context:

### What an Assertion Captures

| Element | Why It Matters |
|---------|---------------|
| **What** was claimed | The actual statement |
| **Who** said it | Plaintiff? Defendant? The contract itself? |
| **When** it was true | January 2023? After the amendment? |
| **Where** it came from | Exact document, page, paragraph |
| **How confident** we are | Contract terms vs. email speculation |

### Example

Instead of storing:
> "Contract value is $5,000,000" ✗

We store:
> "Contract value is $5,000,000"
> — *Source: Amendment 2, Page 3, Paragraph 4*
> — *Effective: June 15, 2023*
> — *Supersedes: Original contract value of $1,000,000*
> — *Confidence: High (executed contract)*

Now when someone asks "What's the contract value?", we can give a complete answer:
- Current value: $5M (per Amendment 2)
- Original value: $1M (superseded)
- Full source trail for verification

---

## What This Enables

### 1. Faster Repeat Queries
Second question about the same matter → use stored assertions instead of re-reading everything.

### 2. Knowledge That Compounds
Information from Question 1 helps answer Question 2. The system gets smarter about each matter over time.

### 3. Contradiction Detection
When two assertions disagree, we can surface that: "Plaintiff claims X, Defendant claims Y" - rather than picking one and hiding the other.

### 4. Full Traceability
Every answer can be traced back to specific documents and pages. Lawyers can verify. Nothing is a black box.

### 5. Temporal Queries
"What did we believe in March?" vs "What do we believe now?" - both are answerable because we track when assertions were valid.

---

## The Challenge We're Working Through

### The Semantic Matching Problem

People say the same thing in different ways:
- "The contract is worth five million dollars"
- "Agreement valued at $5,000,000"
- "Total contract value: $5M"

A human instantly knows these are the same. A computer sees three different text strings.

**We're exploring solutions:**
- Extract structured data (amounts, dates, names) that can be compared precisely
- Use AI similarity matching to group related assertions
- Let humans verify edge cases

This is a hard problem, but it's solvable. We're designing a system that fails gracefully - when matching is uncertain, we preserve both assertions rather than making a wrong merge.

---

## Business Implications

### If We Build This Well

| Metric | Impact |
|--------|--------|
| **Query Speed** | 50%+ faster on repeat queries to same matter |
| **API Costs** | Significant reduction from skipped re-extraction |
| **Answer Quality** | Better because knowledge accumulates |
| **User Trust** | Higher because everything is traceable |
| **Differentiation** | Most legal AI treats each query as isolated |

### What It Requires

- Engineering investment to build the assertion storage layer
- Careful design to avoid the "confident but wrong" failure mode
- Incremental rollout to validate before full commitment

---

## Current Status

We're in the design phase, working through:
1. The exact data structure for assertions
2. How to handle semantic similarity (same meaning, different words)
3. How to integrate with the existing Irys investigation flow
4. What to build first (MVP scope)

No code has been written yet. We're being deliberate because getting this wrong could make the system worse, not better.

---

## The Key Insight

**In legal documents, we're not storing "facts." We're storing "claims made by specific parties at specific times with specific interests" - and if we don't preserve that context, we'll build a system that's confidently wrong.**

The assertion architecture preserves context. It lets us be helpful (faster, smarter) without being dangerous (hiding contradictions, losing provenance, collapsing nuance).

---

## Questions?

Happy to discuss any aspect of this in more detail.
