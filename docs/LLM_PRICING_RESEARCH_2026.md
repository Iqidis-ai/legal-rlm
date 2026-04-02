# LLM Pricing and Capability Research — Irys RLM
**Research Date: April 2026**
**Purpose: Optimize model tier architecture for legal intelligence system**

---

## Executive Summary

The LLM market has shifted significantly. Google's Gemini 3.x series is now in preview alongside stable 2.5 models; OpenAI has introduced GPT-4.1, GPT-5, and a new nano tier; Anthropic has the Claude 4.x line with 1M context windows; and commodity open-weight inference (Groq, Together AI, Fireworks) continues to push budget pricing below $0.10/M tokens. The current Irys RLM tier structure (Flash-Lite → Flash → Pro, all Gemini 2.5) remains sound architecturally but should now be augmented with a NANO tier and the PRO tier should be reconsidered given cost vs. quality alternatives.

---

## Part 1: Comprehensive Pricing Table

### 1.1 Google Gemini (Direct API)

| Model | Context | Input $/M | Output $/M | Batch Discount | Cache Read | Speed (t/s) | Notes |
|---|---|---|---|---|---|---|---|
| gemini-3.1-pro-preview | 1M | $2.00 / $4.00† | $12.00 / $18.00† | 50% | — | 114 | Latest flagship, preview only |
| gemini-3-flash-preview | 1M | $0.50 | $3.00 | 50% | — | 176 | Frontier-class, strong reasoning |
| gemini-3.1-flash-lite-preview | 1M | $0.25 | $1.50 | 50% | — | — | Most cost-efficient Gemini 3.x |
| gemini-2.5-pro | 1M | $1.25 / $2.50† | $10.00 / $15.00† | 50% | 10% of input | — | GA, state-of-the-art |
| gemini-2.5-flash | 1M | $0.30 | $2.50 | 50% | 10% of input | 212 | GA, Irys current FLASH tier |
| gemini-2.5-flash-lite | 1M | **$0.10** | **$0.40** | 50% | 10% of input | — | GA, Irys current LITE tier |
| gemini-2.0-flash | 1M | $0.10 | $0.40 | 50% | 75% discount | — | **DEPRECATED Jun 1 2026** |

† Higher price for prompts >200K tokens.
Cache storage: Flash = $1.00/M tokens/hour; Pro = $4.50/M tokens/hour.

### 1.2 Anthropic Claude (Direct API)

| Model | Context | Input $/M | Output $/M | Batch Input $/M | Batch Output $/M | Cache Hit $/M | Notes |
|---|---|---|---|---|---|---|---|
| Claude Opus 4.6 | 1M | $5.00 | $25.00 | $2.50 | $12.50 | $0.50 | Top tier, 1M ctx at flat rate |
| Claude Opus 4.5 | 200K | $5.00 | $25.00 | $2.50 | $12.50 | $0.50 | — |
| Claude Sonnet 4.6 | 1M | $3.00 | $15.00 | $1.50 | $7.50 | $0.30 | 1M ctx at flat rate |
| Claude Sonnet 4.5 | 200K | $3.00 | $15.00 | $1.50 | $7.50 | $0.30 | — |
| Claude Haiku 4.5 | 200K | $1.00 | $5.00 | $0.50 | $2.50 | $0.10 | — |
| Claude Haiku 3.5 | 200K | $0.80 | $4.00 | $0.40 | $2.00 | $0.08 | Still available |
| Claude Haiku 3 | 200K | $0.25 | $1.25 | $0.125 | $0.625 | $0.03 | Older gen, limited use |

Cache write: 1.25x input (5-min TTL) or 2x input (1-hour TTL). Batch API: 24-hour turnaround, 50% off.

### 1.3 OpenAI (Direct API)

| Model | Context | Input $/M | Output $/M | Batch ~50% off | Notes |
|---|---|---|---|---|---|
| GPT-5.4 | 1.05M | $2.50 | $15.00 | Yes | Frontier flagship |
| GPT-5 / GPT-5.1 | 400K | $0.625 | $5.00 | Yes | Strong mid-tier reasoning |
| GPT-4.1 | 1M | $2.00 | $8.00 | Yes (→$1/$4) | Recommended production, 1M ctx |
| GPT-4.1 Mini | 1M | $0.40 | $1.60 | Yes | Strong mid-tier, 1M ctx |
| GPT-4.1 Nano | 1M | **$0.10** | **$0.40** | Yes | Nano tier, 1M ctx |
| GPT-4o | 128K | $2.50 | $10.00 | Yes | Prior gen flagship |
| GPT-4o-mini | 128K | $0.15 | $0.60 | Yes | Price-perf workhorse |
| o3 | 200K | $2.00 | $8.00 | Yes | Reasoning model |
| o4-mini | 200K | $1.10 | $4.40 | Yes | Budget reasoning |
| GPT-OSS-20B | 131K | $0.03 | $0.10 | — | Cheapest OpenAI offering |
| GPT-OSS-120B | 131K | $0.039 | $0.10 | — | Cheap, decent quality |

Batch API: 50% discount, async processing up to 24 hours.

### 1.4 DeepSeek (Direct API)

| Model | Context | Input $/M (cache miss) | Input $/M (cache hit) | Output $/M | Notes |
|---|---|---|---|---|---|
| deepseek-chat (V3.2) | 128K | $0.28 | $0.028 | $0.42 | Non-thinking mode, excellent value |
| deepseek-reasoner (V3.2) | 128K | $0.28 | $0.028 | $0.42 | Thinking mode (R1-class reasoning) |

Note: DeepSeek R1 (prior gen) was $0.55/$2.19 input/output. Current V3.2 pricing is dramatically cheaper for equivalent quality. Cache hits bring input cost to effectively $0.028/M — competitive with the cheapest models in the market.

### 1.5 Mistral AI (Direct API)

| Model | Context | Input $/M | Output $/M | Notes |
|---|---|---|---|---|
| Mistral Large 3 | 128K | $2.00 | $6.00 | Premium, strong instruction following |
| Mistral Medium 3 | 128K | $1.00 | $3.00 | Balanced tier |
| Mistral Small 3.1 | 128K | $0.20 | $0.60 | Cost-effective, EU-hosted option |
| Mistral Nemo | 128K | $0.15 | $0.15 | Budget, flat rate |

GDPR-compliant EU data residency available. Mistral is a strong alternative for EU legal deployments.

### 1.6 xAI Grok

| Model | Context | Input $/M | Output $/M | Speed (t/s) | Notes |
|---|---|---|---|---|---|
| Grok 4.20 Beta | 2M | $3.00 | $15.00 | 235 | Very fast, 2M context |
| Grok 3 | 131K | $3.00 | $15.00 | — | Production-ready |

Grok 4.20 has a remarkable 2M token context window with 235 t/s — the fastest large context model benchmarked.

### 1.7 Amazon Nova (AWS Bedrock)

| Model | Context | Input $/M | Output $/M | Notes |
|---|---|---|---|---|
| Nova Pro | 300K | $0.80 | $3.20 | Best quality in Nova line |
| Nova Lite | 300K | $0.06 | $0.24 | Good value, multimodal |
| Nova Micro | 128K | **$0.035** | **$0.14** | Cheapest production model |

Requires AWS Bedrock account. Strong for teams already in AWS ecosystem.

### 1.8 Groq (Hosted Inference)

| Model | Context | Input $/M | Output $/M | Speed (t/s) | Notes |
|---|---|---|---|---|---|
| Llama 3.1 8B Instant | 128K | $0.05 | $0.08 | **840** | Fastest available, cheap |
| Llama 4 Scout (17Bx16E) | 128K | $0.11 | $0.34 | 594 | Good quality, very fast |
| GPT-OSS-20B | 128K | $0.075 | $0.30 | 1,000 | Fastest model on Groq |
| GPT-OSS-120B | 128K | $0.15 | $0.60 | 500 | Capable at speed |
| Qwen3 32B | 131K | $0.29 | $0.59 | 662 | Strong reasoning, fast |
| Llama 3.3 70B Versatile | 128K | $0.59 | $0.79 | 394 | Best quality on Groq |

Groq also offers: Batch API (50% discount), Prompt Caching (50% savings on repeated input). These are the fastest inference speeds in the market by a significant margin.

### 1.9 Together AI (Serverless Inference)

| Model | Context | Input $/M | Output $/M | Notes |
|---|---|---|---|---|
| Llama 3 8B Instruct Lite | — | $0.10 | $0.10 | Flat rate |
| Mistral Small 3 | — | $0.10 | $0.30 | Very cheap |
| Qwen3.5 9B | — | $0.10 | $0.15 | Good small model |
| GPT-OSS-120B | — | $0.15 | $0.60 | Capable open model |
| Gemma 3n E4B Instruct | — | $0.02 | $0.04 | Cheapest available |
| Llama 3.3 70B | — | $0.88 | $0.88 | Flat rate, production quality |
| DeepSeek-V3.1 | — | $0.60 | $1.70 | — |
| Kimi K2.5 | 256K | $0.50 | $2.80 | Strong mid-tier |
| DeepSeek-R1-0528 | — | $3.00 | $7.00 | Reasoning, expensive on Together |

Batch API available for ~50% discount on most models.

### 1.10 Cohere

| Model | Context | Input $/M | Output $/M | Notes |
|---|---|---|---|---|
| Command R+ (08-2024) | 128K | $2.50 | $10.00 | Best quality, enterprise RAG |
| Command R (03-2024) | 128K | $0.50 | $1.50 | Mid-tier |
| Command Light | — | $0.30 | $0.60 | Budget |
| Aya Expanse 8B/32B | — | $0.50 | $1.50 | 23-language multilingual |

Cohere differentiates on enterprise RAG tooling (Rerank, Embed) rather than raw generation. Less relevant for pure generation workloads.

### 1.11 Emerging/Notable Models

| Model | Provider | Context | Input $/M | Output $/M | Intelligence Score | Notes |
|---|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview | Google | 1M | $2.00 | $12.00 | 57 (tied #1) | Latest, preview |
| Llama 4 Scout (via Together) | Meta/Together | 10M | ~$0.20 | ~$0.30 | — | **10M context window** |
| Llama 4 Maverick | Meta/Fireworks | 1M | ~$0.22 | ~$0.55 | — | 1M context, open |
| Qwen3.5 397B A17B | Alibaba | 262K | ~$0.60 | ~$2.40 | 45 | Strong reasoning MoE |
| MiniMax M2.7 | MiniMax | 205K | — | — | 50 | Surprise entrant, $0.53 blended |
| GLM-5 | Z AI | 200K | ~$1.00 | ~$3.20 | 50 | Competitive intelligence |
| DeepSeek V3.2 reasoner | DeepSeek | 128K | $0.28 | $0.42 | — | R1-class, fraction of cost |

---

## Part 2: Quality Tier Positioning

### NANO Tier (≤$0.10/M input, production-viable)
Models where raw speed and volume matter most; quality is acceptable for extraction/triage.

| Model | Input $/M | Output $/M | Strengths | Weaknesses |
|---|---|---|---|---|
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | 1M context, structured output, GA stable, 90% cache discount | Weakest reasoning of Gemini 2.5 family |
| GPT-4.1 Nano | $0.10 | $0.40 | 1M context, strong instruction following | OpenAI pricing risk |
| Amazon Nova Micro | $0.035 | $0.14 | Cheapest production model | AWS lock-in, 128K context only |
| Groq Llama 3.1 8B | $0.05 | $0.08 | 840 t/s — fastest in class | 128K context, quality ceiling |
| DeepSeek V3.2 (cached) | $0.028 | $0.42 | Near-zero input cost with cache | 128K context, mainland China routing |
| Gemma 3n E4B (Together) | $0.02 | $0.04 | Cheapest by far | Small model, limited capability |

**Recommendation for NANO:** `gemini-2.5-flash-lite` is the strongest choice — it is GA, has 1M context, structured output, and the 90% cache discount for repeated document sections makes it exceptionally cheap in practice.

### LITE Tier ($0.10–$0.60/M input, bulk analysis)
Solid extraction, decent reasoning, structured JSON output reliable.

| Model | Input $/M | Output $/M | Strengths |
|---|---|---|---|
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | Current Irys LITE — excellent |
| GPT-4.1 Mini | $0.40 | $1.60 | Strong instruction following, 1M context |
| Mistral Small 3.1 | $0.20 | $0.60 | EU-hosted, GDPR, good at extraction |
| Groq Llama 4 Scout | $0.11 | $0.34 | 594 t/s, solid quality |
| Amazon Nova Lite | $0.06 | $0.24 | Very cheap, multimodal |
| DeepSeek V3.2 (cache miss) | $0.28 | $0.42 | Excellent overall quality for price |

### FLASH Tier ($0.30–$2.00/M input, reasoning + analysis)
Good reasoning, reliable structured output, multi-step analysis.

| Model | Input $/M | Output $/M | Speed | Strengths |
|---|---|---|---|---|
| Gemini 2.5 Flash | $0.30 | $2.50 | 212 t/s | Current Irys FLASH — strong |
| Gemini 3 Flash Preview | $0.50 | $3.00 | 176 t/s | Frontier-class, preview |
| GPT-4.1 | $2.00 | $8.00 | — | 1M context, excellent JSON |
| GPT-4o-mini | $0.15 | $0.60 | — | Cheap, reliable |
| Claude Haiku 3.5 | $0.80 | $4.00 | — | Strong Anthropic quality at mid price |
| DeepSeek V3.2 reasoner | $0.28 | $0.42 | — | R1-class reasoning at NANO price |

### PRO Tier ($1.25+/M input, final synthesis)
Highest quality, coherent long-form output, complex multi-issue reasoning.

| Model | Input $/M | Output $/M | Context | Intelligence Score | Strengths |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | $1.25 | $10.00 | 1M | — | Current Irys PRO — strong |
| Gemini 3.1 Pro Preview | $2.00 | $12.00 | 1M | 57 (tied #1) | Highest intelligence score benchmarked |
| Claude Opus 4.6 | $5.00 | $25.00 | 1M | 46–53 | Best for legal prose, belief-revision tasks |
| Claude Sonnet 4.6 | $3.00 | $15.00 | 1M | 44–52 | Best cost/quality balance in PRO tier |
| GPT-4.1 | $2.00 | $8.00 | 1M | — | Strongest structured output in PRO tier |
| GPT-5 | $0.625 | $5.00 | 400K | 45 | Remarkable price for intelligence score |
| Grok 4.20 Beta | $3.00 | $15.00 | **2M** | 48 | Only 2M context window; fastest at 235 t/s |

---

## Part 3: Batch API Discounts Summary

| Provider | Batch Discount | Turnaround | Applicable Models |
|---|---|---|---|
| Google Gemini API | 50% off all rates | 24h | All Gemini models |
| Anthropic | 50% off all rates | 24h async | All Claude models |
| OpenAI | 50% off all rates | 24h | Most GPT and o-series models |
| Groq | 50% off all rates | 24h–7 days | All hosted models |
| Together AI | ~50% off | Variable | Most serverless models |
| DeepSeek | Native prompt cache | Real-time | Cache miss→hit pricing |

**For Irys RLM:** Document ingestion pipelines (cold-path processing of new document batches) should use Batch API wherever latency is not critical. This alone cuts cold-path costs by 50%.

---

## Part 4: Context Window Reference

| Context Size | Models |
|---|---|
| 10M tokens | Llama 4 Scout (via Together/Fireworks) |
| 2M tokens | Grok 4.20 Beta |
| 1.05M tokens | GPT-5.4 |
| 1M tokens | Gemini 2.5 Flash-Lite, Gemini 2.5 Flash, Gemini 2.5 Pro, Gemini 3.x series, GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano, Claude Opus/Sonnet 4.6 |
| 400K tokens | GPT-5 family |
| 200K tokens | Claude Haiku 4.5, Claude Opus/Sonnet 4.5, o3, o4-mini, Grok 3, GLM-5 |
| 128K–131K tokens | DeepSeek V3.2, Mistral models, GPT-4o, most Groq models |

**For Irys RLM legal work:** 1M context is now the practical minimum for the FLASH and PRO tiers. A 500-page legal record is ~375K tokens. A full litigation file with exhibits can easily exceed 500K tokens. The Gemini 2.5 line remains excellent here.

---

## Part 5: Cost Estimates — 1,000 Queries/Day

### Query Token Profile Assumptions

| Query Type | Input Tokens | Output Tokens | Rationale |
|---|---|---|---|
| NANO pass (doc triage/extraction) | 8,000 | 500 | Single document section read |
| LITE pass (bulk document read) | 15,000 | 2,000 | Full document read + extraction |
| FLASH pass (analysis/orientation) | 25,000 | 4,000 | Multi-doc analysis, structured JSON |
| PRO pass (final synthesis) | 40,000 | 8,000 | Full matter synthesis, long-form memo |

### Scenario A: Current Irys Setup (Gemini 2.5 all-tiers)
*Assuming 1,000 queries/day split: 500 LITE + 350 FLASH + 150 PRO*

| Tier | Model | Input $/M | Output $/M | Daily Input Tokens | Daily Output Tokens | Daily Cost |
|---|---|---|---|---|---|---|
| LITE (500 queries) | gemini-2.5-flash-lite | $0.10 | $0.40 | 7.5M | 1.0M | $0.75 + $0.40 = **$1.15** |
| FLASH (350 queries) | gemini-2.5-flash | $0.30 | $2.50 | 8.75M | 1.40M | $2.63 + $3.50 = **$6.13** |
| PRO (150 queries) | gemini-2.5-pro | $1.25 | $10.00 | 6.0M | 1.20M | $7.50 + $12.00 = **$19.50** |
| **TOTAL** | | | | **22.25M** | **3.60M** | **$26.78/day = ~$803/month** |

### Scenario B: Optimized — Add NANO tier, keep Gemini for FLASH/PRO, use Claude for premium
*Split: 400 NANO + 300 LITE + 200 FLASH + 100 PRO*

| Tier | Model | Daily Cost |
|---|---|---|
| NANO (400 queries) | gemini-2.5-flash-lite | 400×(8K×$0.10/M + 0.5K×$0.40/M) = $0.32 + $0.08 = **$0.40** |
| LITE (300 queries) | gemini-2.5-flash-lite | 300×(15K×$0.10/M + 2K×$0.40/M) = $0.45 + $0.24 = **$0.69** |
| FLASH (200 queries) | gemini-2.5-flash | 200×(25K×$0.30/M + 4K×$2.50/M) = $1.50 + $2.00 = **$3.50** |
| PRO (100 queries) | claude-sonnet-4.6 | 100×(40K×$3.00/M + 8K×$15.00/M) = $12.00 + $12.00 = **$24.00** |
| **TOTAL** | | **$28.59/day = ~$858/month** |

### Scenario C: Most Aggressive Cost Reduction
*Split: 500 NANO + 300 LITE + 150 FLASH + 50 PRO — using DeepSeek for LITE/FLASH, Gemini PRO*

| Tier | Model | Daily Cost |
|---|---|---|
| NANO (500 queries) | gemini-2.5-flash-lite | $0.40 + $0.10 = **$0.50** |
| LITE (300 queries) | deepseek-chat V3.2 | 300×(15K×$0.28/M + 2K×$0.42/M) = $1.26 + $0.25 = **$1.51** |
| FLASH (150 queries) | deepseek-reasoner V3.2 | 150×(25K×$0.28/M + 4K×$0.42/M) = $1.05 + $0.25 = **$1.30** |
| PRO (50 queries) | gemini-2.5-pro | 50×(40K×$1.25/M + 8K×$10.00/M) = $2.50 + $4.00 = **$6.50** |
| **TOTAL** | | **$9.81/day = ~$294/month** |

*Note: DeepSeek's mainland China routing raises data sovereignty concerns for legal matter content. Not recommended without client approval.*

### Scenario D: Batch API — 50% savings on cold-path processing
Apply 50% Batch API discount to all NANO and LITE tier work (cold-path document ingestion is inherently non-realtime):

| Tier | Effective Rate | Saving vs. Scenario A |
|---|---|---|
| NANO batch | 50% off | Significant for high-volume ingestion |
| LITE batch | 50% off | ~$0.50/day → ~$0.25/day |
| FLASH realtime | No discount | — |
| PRO realtime | No discount | — |

**Practical estimate for Scenario A with batch on NANO+LITE: ~$19–22/day = ~$580–660/month**

### Summary Table

| Scenario | Models | Monthly Cost | Notes |
|---|---|---|---|
| A: Current Setup | Gemini 2.5 all tiers | ~$803 | Baseline |
| A+Batch: Current + batch on LITE | Gemini 2.5 all tiers | ~$630 | Easy win |
| B: Add NANO, Claude PRO | Gemini + Claude | ~$858 | Higher PRO quality |
| C: Aggressive + DeepSeek | Mixed | ~$294 | Data sovereignty risk |
| D: Gemini all + batch LITE | Gemini 2.5 all tiers | ~$620 | Recommended near-term |

---

## Part 6: Recommended Tier Mappings for Irys RLM

### Proposed Architecture: 4-Tier System

```
NANO  →  gemini-2.5-flash-lite          ($0.10 in / $0.40 out)
LITE  →  gemini-2.5-flash-lite          ($0.10 in / $0.40 out, larger context)
FLASH →  gemini-2.5-flash               ($0.30 in / $2.50 out)
PRO   →  gemini-2.5-pro                 ($1.25 in / $10.00 out)
```

**Or with upgrade candidates:**
```
NANO  →  gemini-2.5-flash-lite          Bulk triage, doc type classification
LITE  →  gemini-2.5-flash-lite          Full document reading + assertion extraction
FLASH →  gemini-2.5-flash               Analysis, orientation, issue mapping
PRO   →  gemini-3.1-pro-preview         Final synthesis (if preview stability acceptable)
         OR gemini-2.5-pro              Final synthesis (GA, stable)
         OR claude-sonnet-4.6           Final synthesis (if prose quality priority)
```

### Tier Mapping by Task

| Task | Recommended Tier | Model | Rationale |
|---|---|---|---|
| Document type classification | NANO | gemini-2.5-flash-lite | Cheap, 1M ctx, fast |
| Bulk text extraction (OCR pass) | NANO | gemini-2.5-flash-lite | High volume, simple |
| Initial actor/entity extraction | NANO | gemini-2.5-flash-lite | Structured JSON, cheap |
| Full document card creation | LITE | gemini-2.5-flash-lite | Larger output, still cheap |
| Assertion extraction with speech-act typing | LITE | gemini-2.5-flash-lite | JSON schema, 1M ctx |
| Gap detection (what's missing?) | FLASH | gemini-2.5-flash | Reasoning required |
| Issue model mapping | FLASH | gemini-2.5-flash | Multi-step reasoning |
| Source-role calibration | FLASH | gemini-2.5-flash | Nuanced judgment |
| Quantitative extraction + reconciliation | FLASH | gemini-2.5-flash | Structured + reasoning |
| Final synthesis memo | PRO | gemini-2.5-pro | Coherent long-form |
| Legal research (authority analysis) | PRO | gemini-2.5-pro | Complex legal reasoning |
| Belief revision cascade | PRO | gemini-2.5-pro | Dependency graph traversal |
| Adversarial analysis | PRO | claude-sonnet-4.6 (alt) | Best for adversarial framing |

### Splitting LITE vs. NANO

The current Irys architecture combines NANO and LITE into one tier (gemini-2.5-flash-lite). Given the model costs the same for both, the operational distinction is token budget rather than model:

- **NANO budget**: ~500–2,000 output tokens. Use for classification, triage, entity spotting.
- **LITE budget**: ~2,000–8,000 output tokens. Use for full extraction tasks.

This means the "NANO tier" is a task-routing decision, not necessarily a different model. The practical addition is: route trivial triage tasks to shorter-context, shorter-output calls on flash-lite rather than ever sending them to the FLASH or PRO models.

---

## Part 7: Special Capabilities Matrix

### Structured Output / JSON Schema Support

| Provider | Native JSON Mode | Schema-enforced | Tool Calling | Reliability |
|---|---|---|---|---|
| Google Gemini 2.5+ | Yes | Yes (responseSchema) | Yes | Excellent |
| Google Gemini 3.x | Yes | Yes | Yes + Computer Use | Excellent |
| Anthropic Claude 4.x | Yes | Yes (tool_use mode) | Yes | Excellent |
| OpenAI GPT-4.1+ | Yes | Yes (Structured Outputs) | Yes | Excellent |
| DeepSeek V3.2 | Yes | Yes (JSON output) | Yes | Good |
| Mistral models | Yes | Partial | Yes | Good |
| Groq-hosted models | Partial | Via prompt | Limited | Variable |

### Long-Context Document Analysis

| Capability | Best Model | Notes |
|---|---|---|
| 10M token context | Llama 4 Scout (Together) | Open weight, routing risk |
| 2M token context | Grok 4.20 Beta | Fast, xAI routing |
| 1M token context (GA, proprietary) | Gemini 2.5 family, Claude Opus/Sonnet 4.6, GPT-4.1 | All solid choices |
| Context caching (Google) | Gemini 2.5+ | 90% discount on cache reads, hourly storage |
| Prompt caching (Anthropic) | Claude 4.x | 10% of input price for cache hits |

### Speed (Tokens/Second) — Production Inference

| Model | t/s | Use Case |
|---|---|---|
| Groq GPT-OSS-20B | ~1,000 | Fastest available, low quality ceiling |
| Groq Llama 3.1 8B | 840 | Fast + cheap for bulk triage |
| Grok 4.20 Beta | 235 | Fast frontier model |
| Gemini 3 Flash | 176 | Fast frontier class |
| Gemini 2.5 Flash | 212 | GA stable, fast |
| GPT-5 Nano | ~194 | OpenAI cheap + fast |
| Claude Opus 4.6 | ~42 | Slow but highest quality |
| Claude Sonnet 4.6 | ~43–45 | Moderate speed |

---

## Part 8: Models Worth Adding as a NANO Tier

### Recommendation: No Additional Model Required

`gemini-2.5-flash-lite` at $0.10/$0.40 per million tokens (input/output) is production-grade and already serves as Irys's LITE tier. For a NANO tier, the primary change is **task routing** (shorter prompts, narrower extraction scope), not a different model.

### If a Cheaper NANO Model Is Warranted

These are the candidates genuinely below $0.10/M input that are production-viable:

| Model | Input $/M | Output $/M | Verdict |
|---|---|---|---|
| Amazon Nova Micro | $0.035 | $0.14 | Good for AWS shops; 128K ctx is limiting |
| Groq Llama 3.1 8B | $0.05 | $0.08 | Excellent for speed; not great at structured JSON at scale |
| Gemma 3n E4B (Together) | $0.02 | $0.04 | Cheapest; small model, limited capability |
| DeepSeek V3.2 (cached input) | $0.028 | $0.42 | Excellent if data sovereignty is not a concern |

**Verdict:** For Irys RLM, where legal matter data sensitivity is paramount, DeepSeek (China routing) and most third-party open-weight providers should not handle raw matter content without careful legal review. `gemini-2.5-flash-lite` is already near-optimal for the NANO/LITE use case within a jurisdiction-safe provider (Google). Adding a separate NANO model adds operational complexity without significant cost benefit at typical Irys query volumes.

---

## Part 9: Key Findings and Recommendations

### Immediate Actions

1. **Add NANO tier as a task-routing concept, not a new model.** Route triage/classification calls (document type, actor spotting, section headers) to short-budget calls on `gemini-2.5-flash-lite`. This will reduce the average token consumption per LITE-tier call significantly.

2. **Enable Google Batch API for cold-path ingestion.** All document ingestion (not user-interactive queries) should use the Batch API for 50% cost reduction with no quality tradeoff and only 24-hour turnaround. At 1,000 queries/day this saves ~$150–200/month.

3. **Enable context caching on Gemini 2.5 Flash/Flash-Lite.** Legal matters have stable document corpora. System prompts, matter model context, and extracted document summaries repeated across calls get 90% cost reduction on cache hits. At 1M token context usage, storage costs $1.00/hour but cache reads cost only $0.01/M vs. $0.10/M.

4. **Do not migrate off Gemini 2.5 to Gemini 3.x preview models.** The 3.x series is in preview and does not have GA SLAs. For a production legal system, stability > marginal quality improvement. Revisit when GA.

5. **Consider Claude Sonnet 4.6 as a PRO tier alternative for final synthesis.** At $3.00/$15.00 vs. Gemini 2.5 Pro's $1.25/$10.00, it is more expensive but consistently produces better structured legal reasoning prose and handles adversarial framing well. For high-stakes synthesis output (partner-facing memos), the quality premium may be justified. Can be A/B tested against Gemini 2.5 Pro.

6. **Gemini 2.0 Flash will be deprecated June 1, 2026.** If any legacy code still references `gemini-2.0-flash`, migrate to `gemini-2.5-flash-lite` immediately.

### Model Stability Risk Assessment

| Model | Stability Risk | Notes |
|---|---|---|
| gemini-2.5-flash-lite | Low | GA stable |
| gemini-2.5-flash | Low | GA stable |
| gemini-2.5-pro | Low | GA stable |
| gemini-3.x series | High | Preview only, no SLA |
| claude-sonnet-4.6 | Low | GA |
| deepseek-chat V3.2 | Medium | Provider stability, data routing |
| GPT-4.1 | Low | GA |
| GPT-5.x | Medium | Some variants preview |

### Cost vs. Quality Summary

For Irys RLM's three core task categories:

**Bulk document reading (extraction, assertion, entity):**
- Best value: `gemini-2.5-flash-lite` — $0.10/$0.40, 1M context, structured JSON, GA
- Batch API makes this ~$0.05/$0.20 for non-realtime ingestion

**Analysis/reasoning (orientation, issue mapping, gap detection):**
- Best value: `gemini-2.5-flash` — $0.30/$2.50, 212 t/s, strong reasoning
- Alternative if budget critical: `deepseek-chat V3.2` — $0.28/$0.42 (similar input cost, dramatically cheaper output), but 128K context only

**Final synthesis (partner memos, legal analysis):**
- Best value: `gemini-2.5-pro` — $1.25/$10.00, 1M context, GA, current Irys PRO
- Best quality: `claude-sonnet-4.6` — $3.00/$15.00, 1M context, better prose and adversarial reasoning
- Best price/intelligence: `GPT-5` — $0.625/$5.00 (intelligence score 45), but only 400K context

---

## Part 10: Quick Reference Cheat Sheet

```
PROVIDER | MODEL                    | IN $/M  | OUT $/M | CTX    | TIER
---------|--------------------------|---------|---------|--------|-------
Google   | gemini-3.1-pro-preview   | $2.00   | $12.00  | 1M     | PRO+
Google   | gemini-3-flash-preview   | $0.50   | $3.00   | 1M     | FLASH+
Google   | gemini-2.5-pro           | $1.25   | $10.00  | 1M     | PRO     ← Irys PRO
Google   | gemini-2.5-flash         | $0.30   | $2.50   | 1M     | FLASH   ← Irys FLASH
Google   | gemini-2.5-flash-lite    | $0.10   | $0.40   | 1M     | LITE    ← Irys LITE
Anthropic| claude-opus-4.6          | $5.00   | $25.00  | 1M     | PRO+
Anthropic| claude-sonnet-4.6        | $3.00   | $15.00  | 1M     | PRO
Anthropic| claude-haiku-4.5         | $1.00   | $5.00   | 200K   | FLASH
Anthropic| claude-haiku-3.5         | $0.80   | $4.00   | 200K   | FLASH
OpenAI   | gpt-5                    | $0.625  | $5.00   | 400K   | PRO
OpenAI   | gpt-4.1                  | $2.00   | $8.00   | 1M     | PRO/FLASH
OpenAI   | gpt-4.1-mini             | $0.40   | $1.60   | 1M     | FLASH
OpenAI   | gpt-4.1-nano             | $0.10   | $0.40   | 1M     | LITE
OpenAI   | gpt-4o-mini              | $0.15   | $0.60   | 128K   | LITE
OpenAI   | o3                       | $2.00   | $8.00   | 200K   | PRO (reasoning)
OpenAI   | o4-mini                  | $1.10   | $4.40   | 200K   | FLASH (reasoning)
DeepSeek | deepseek-chat V3.2       | $0.28   | $0.42   | 128K   | FLASH
DeepSeek | deepseek-reasoner V3.2   | $0.28   | $0.42   | 128K   | FLASH (reasoning)
Mistral  | mistral-large-3          | $2.00   | $6.00   | 128K   | PRO
Mistral  | mistral-small-3.1        | $0.20   | $0.60   | 128K   | FLASH
xAI      | grok-4.20-beta           | $3.00   | $15.00  | 2M     | PRO (fast)
Amazon   | nova-micro               | $0.035  | $0.14   | 128K   | NANO
Amazon   | nova-lite                | $0.06   | $0.24   | 300K   | NANO/LITE
Groq     | llama-3.1-8b-instant     | $0.05   | $0.08   | 128K   | NANO (fast)
Groq     | llama-3.3-70b-versatile  | $0.59   | $0.79   | 128K   | FLASH (fast)
```

---

*Sources consulted: Google AI Developer Pricing, Anthropic Pricing Docs, OpenAI Pricing, DeepSeek API Docs, Mistral AI Pricing, Groq Pricing, Together AI Pricing, Cohere Pricing, Artificial Analysis LLM Leaderboard, pricepertoken.com, multiple third-party pricing aggregators. All prices as of April 2026. Verify before committing to production budgets — pricing changes frequently.*
