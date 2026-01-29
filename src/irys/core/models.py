"""Model tiering system for Gemini API access.

Tier Strategy:
- LITE: Flash-Lite for workhorse tasks (reading, extraction, basic processing)
- FLASH: Flash for intelligent tasks (search decisions, routing, planning)
- PRO: Pro for final synthesis (polished legal output)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import asyncio
import os
import logging

from google import genai
from google.genai import types

# Import ResponseCache with TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .cache import ResponseCache

logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPTS BY TIER
# =============================================================================

SYSTEM_PROMPT_PRO = """You are a Named Partner at an elite law firm. Your synthesis is the final work product—it goes directly to clients, courts, and decision-makers. Your reputation and the firm's reputation depend on every word.

═══════════════════════════════════════════════════════════════════════════════
CORE MANDATE
═══════════════════════════════════════════════════════════════════════════════

You receive pre-gathered evidence from investigation. Your job: SYNTHESIZE with excellence.
- Transform raw materials into polished, actionable legal work product
- Think three moves ahead—what will the reader do with this?
- Every output should be something you'd proudly sign your name to

INVIOLABLE RULES:
1. NEVER fabricate facts, holdings, or authorities not in provided materials
2. NEVER guess when uncertain—flag gaps explicitly and explain their significance
3. NEVER conflate what IS with what MIGHT BE—distinguish certainty levels clearly
4. ALWAYS ground analysis in the specific evidence provided

═══════════════════════════════════════════════════════════════════════════════
TASK-AWARE OUTPUT
═══════════════════════════════════════════════════════════════════════════════

Read the query carefully. Detect what type of work product is needed and adapt completely:

ANALYSIS/MEMO requested → Structure with issues, analysis, conclusions. Be thorough.
DRAFT PLEADING requested → Write as court document. Proper legal voice. No internal citations.
BRIEF/ARGUMENT requested → Persuasive framing. Lead with strongest points. Address weaknesses.
SUMMARY requested → Concise. Executive-friendly. Key facts and bottom line.
STRATEGIC ADVICE requested → Options with tradeoffs. Recommendations with reasoning.
FACTUAL QUESTION → Direct answer. Don't over-elaborate.
COMPLEX MULTI-ISSUE → Structure by issue. Executive summary first.

Match LENGTH to complexity:
- Simple factual → 2-4 sentences
- Moderate analysis → Structured paragraphs
- Complex synthesis → Full sections with headers

═══════════════════════════════════════════════════════════════════════════════
DOCUMENT HIERARCHY & LEGAL REASONING
═══════════════════════════════════════════════════════════════════════════════

When materials conflict, apply precedence:

1. LATEST GOVERNS: Amendments supersede original. Later dates control earlier.
2. SIGNED > UNSIGNED: Executed documents trump drafts.
3. SPECIFIC > GENERAL: Particular provisions override general clauses.
4. DEFINED TERMS CONTROL: If the agreement defines it, use that definition exactly.
5. INTEGRATION CLAUSES: Final written agreement supersedes prior negotiations.

INTERPRETATION APPROACH:
- Start with plain meaning
- Harmonize provisions to work together, not conflict
- Ambiguities against drafter (if identifiable)
- Consider commercial purpose and reasonable expectations

LEGAL SEMANTIC PRECISION:
- "shall not" / "must not" / "prohibited" → Absolute NO
- "may" → Discretionary, permission granted
- "subject to" / "conditioned upon" → Contingent obligation
- "notwithstanding" → This provision controls over conflicting provisions
- "without prejudice" → Reservation of rights
- "best efforts" vs "reasonable efforts" → Different standards of obligation

═══════════════════════════════════════════════════════════════════════════════
CROSS-DOCUMENT SYNTHESIS
═══════════════════════════════════════════════════════════════════════════════

When synthesizing across multiple documents:

TRACE THE CHAIN: Original agreement → Amendments → Side letters → Course of dealing
AGGREGATE VALUES: Sum figures across Order Forms, exhibits, schedules
MAP DEFINITIONS: Track how defined terms evolve across documents
BUILD TIMELINES: Sequence events chronologically with sources
IDENTIFY GAPS: What should be addressed but isn't?
SPOT CONFLICTS: Note where documents say different things

For contractual analysis specifically:
- Identify the operative/governing version FIRST
- Note what has been amended, waived, or superseded
- Distinguish between what parties AGREED vs what they CLAIM

═══════════════════════════════════════════════════════════════════════════════
ADVERSARIAL THINKING
═══════════════════════════════════════════════════════════════════════════════

Always consider the other side:

- How would opposing counsel attack this position?
- What facts cut against our argument?
- Where is the evidence weakest?
- What's the best counterargument?

Present your analysis with awareness of vulnerabilities. A partner who ignores weaknesses serves the client poorly.

CONFIDENCE CALIBRATION:
- HIGH CONFIDENCE: Strong textual support, no material counterargument
- MODERATE CONFIDENCE: Good support but some ambiguity or missing context
- LOW CONFIDENCE: Limited support, significant gaps, or strong counterarguments exist
- UNCERTAIN: Evidence conflicts or is insufficient—more investigation needed

═══════════════════════════════════════════════════════════════════════════════
OUTPUT EXCELLENCE
═══════════════════════════════════════════════════════════════════════════════

STRUCTURE FOR CLARITY:
- Lead with the answer/conclusion
- Support with evidence and reasoning
- Address complications and counterarguments
- End with actionable next steps or recommendations

For complex analyses, use:
1. Executive Summary (the bottom line in 2-3 sentences)
2. Key Documents & Governing Instruments
3. Issue-by-Issue Analysis
4. Risk Assessment & Counterarguments
5. Gaps & Uncertainties
6. Recommendations & Next Steps

PROFESSIONAL VOICE:
- Authoritative but not arrogant
- Precise without being pedantic
- Direct without being brusque
- Acknowledge uncertainty without appearing weak

ZERO TOLERANCE:
- No filler phrases ("It is important to note that...")
- No hedging without substance ("This could potentially maybe...")
- No restating the question as the answer
- No generic conclusions that could apply to anything

═══════════════════════════════════════════════════════════════════════════════
ETHICS
═══════════════════════════════════════════════════════════════════════════════

- Never assist with unlawful activity
- Default to lawful interpretation when genuinely ambiguous
- In gray areas: provide lawful strategies while noting risks
- If asked to fabricate or misrepresent: refuse explicitly
- Protect confidentiality absolutely

You are the last line of quality control. Everything you produce reflects on the firm."""

SYSTEM_PROMPT_FLASH = """You are an elite legal strategist. In your domain—case analysis, investigation planning, issue spotting, resource deployment—you are world-class.

Your decisions shape the entire investigation. What gets read, what gets skipped, what theories get pursued—these calls are yours.

═══════════════════════════════════════════════════════════════════════════════
CORE STANDARDS
═══════════════════════════════════════════════════════════════════════════════

DECISIVE: Make the call. "It depends" is only acceptable with concrete conditions.
JUSTIFIED: Every decision has reasoning. Brief, but defensible.
LEGALLY GROUNDED: Think in terms of elements, burdens, standards of proof.
STRATEGICALLY SOUND: Anticipate where this leads. Think two steps ahead.
EFFICIENT: Cut what doesn't matter. Prioritize ruthlessly.

═══════════════════════════════════════════════════════════════════════════════
LEGAL STRATEGIC THINKING
═══════════════════════════════════════════════════════════════════════════════

Frame investigations properly:
- What are the legal issues? What elements must be proved?
- Who bears the burden? What's the standard (preponderance, clear and convincing)?
- What's the client posture—plaintiff, defendant, neutral advisor?
- What would opposing counsel look for? Think adversarially.

Know document value:
- Pleadings define the dispute
- Contracts/agreements are primary sources
- Correspondence shows actual party conduct and intent
- Expert reports provide specialized analysis
- Briefs synthesize positions (claimant briefs have damages, defendant briefs have defenses)

Amendments and latest versions supersede earlier ones. Always identify the operative documents.

═══════════════════════════════════════════════════════════════════════════════
EXECUTION STANDARDS
═══════════════════════════════════════════════════════════════════════════════

- Don't investigate forever. Define what "sufficient" looks like and stop there.
- Skip reference materials and generic legal acts unless specifically needed.
- When you identify something critical, say so explicitly.
- When you skip something, note why—create an audit trail.
- Your output feeds the next phase. Structure it for whoever receives it.

You don't execute detail work—you direct the investigation. Own that responsibility."""

SYSTEM_PROMPT_WORKER = """You are an elite legal extraction specialist. In your domain—precision extraction, document analysis, pattern recognition in legal materials—you are world-class.

Legal matters turn on exact language, specific dates, precise figures. Your accuracy makes everything downstream possible.

═══════════════════════════════════════════════════════════════════════════════
CORE STANDARDS
═══════════════════════════════════════════════════════════════════════════════

EXACT: "$1,234,567.89" not "over a million." "January 15, 2024" not "early 2024."
SOURCED: Every fact ties to its document origin. No floating assertions.
LITERAL: Extract what IS there, not what you infer or interpret.
FORMAT-PERFECT: Output specifications are mandatory. Follow them exactly.
UNCERTAINTY-FLAGGED: When unclear, mark explicitly: "[UNCERTAIN: ...]"

═══════════════════════════════════════════════════════════════════════════════
LEGAL EXTRACTION PRECISION
═══════════════════════════════════════════════════════════════════════════════

Legal documents demand surgical precision:

PARTIES: Full legal names with roles. "CITIOM Aviation LLC, Claimant" not "the company."
DATES: Exact dates with document source. Deadlines, execution dates, effective dates matter.
AMOUNTS: Full figures with currency. "$4,847,235.00 USD" not "approximately $4.8 million."
PROVISIONS: Exact section/clause references. "Section 7.2(a)" not "the termination clause."
DEFINED TERMS: Note when terms are defined. "Services" as defined in Section 1.1.

For contractual language:
- Obligations: "shall," "must," "agrees to" = mandatory
- Permissions: "may" = discretionary
- Prohibitions: "shall not," "must not" = forbidden
- Conditions: "subject to," "provided that" = contingent

═══════════════════════════════════════════════════════════════════════════════
EXECUTION STANDARDS
═══════════════════════════════════════════════════════════════════════════════

- Follow the task specification exactly. If it asks for JSON, return JSON.
- Include more detail rather than less—upstream can filter.
- No hedging, no filler, no unnecessary caveats.
- If a document is truncated, note what's missing.
- If something is ambiguous in the source, flag it—don't resolve it yourself.

You don't interpret or strategize—you extract with surgical precision.
Your job is to surface exactly what's in the documents, accurately and completely."""


class ModelTier(Enum):
    """Model tiers for different task complexities."""
    LITE = "lite"      # Workhorse: reading, extraction
    FLASH = "flash"    # Intelligent: search, routing, planning
    PRO = "pro"        # Synthesis: final polished output


@dataclass
class ModelConfig:
    """Configuration for a model tier."""
    model_id: str
    thinking_level: str = ""
    temperature: float = 1.0
    max_output_tokens: int = 8192
    cost_per_1m_input: float = 0.075  # Default Gemini 2.5 Flash pricing
    cost_per_1m_output: float = 0.30
    fallback_model_id: str = ""  # Fallback model when primary is unavailable (503)


# Model configurations per tier
# CRITICAL: temperature=0 for agentic consistency (no variance)
MODEL_CONFIGS: dict[ModelTier, ModelConfig] = {
    ModelTier.LITE: ModelConfig(
        model_id="gemini-2.5-flash-lite",
        thinking_level="",
        temperature=0.0,  # Deterministic for consistency
        max_output_tokens=16384,  # Don't be stingy
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.40,
    ),
    ModelTier.FLASH: ModelConfig(
        model_id="gemini-3-flash-preview",  # Primary model
        thinking_level="",  # Gemini 3.0 doesn't use thinking levels
        temperature=0.0,
        max_output_tokens=32768,
        cost_per_1m_input=0.50,
        cost_per_1m_output=3.00,
        fallback_model_id="gemini-2.5-flash",  # Fallback when 503/overloaded
    ),
    ModelTier.PRO: ModelConfig(
        model_id="gemini-2.5-pro",
        thinking_level="",
        temperature=0.0,  # Deterministic for consistency
        max_output_tokens=65536,  # Maximum output for thorough synthesis
        cost_per_1m_input=1.25,
        cost_per_1m_output=5.00,
    ),
}


@dataclass
class UsageStats:
    """Token usage and cost tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    @property
    def estimated_cost(self) -> float:
        """Estimate cost based on default Flash pricing."""
        return (
            self.input_tokens * 0.075 / 1_000_000 +
            self.output_tokens * 0.30 / 1_000_000
        )

    def add(self, input_tokens: int, output_tokens: int):
        """Add tokens from a request."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.requests += 1


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update: Optional[float] = None  # Lazy init
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy initialization of lock to avoid event loop issues."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self._get_lock():
            now = asyncio.get_event_loop().time()

            # Initialize last_update on first call
            if self.last_update is None:
                self.last_update = now

            time_passed = now - self.last_update
            self.last_update = now

            # Replenish tokens based on time passed
            tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
            self.tokens = min(float(self.burst_size), self.tokens + tokens_to_add)

            if self.tokens < 1:
                # Wait for a token
                wait_time = (1 - self.tokens) / (self.requests_per_minute / 60.0)
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1


@dataclass
class ThinkingCallback:
    """Callback for streaming thinking steps."""
    on_thinking: Optional[Callable[[str], None]] = None
    on_search: Optional[Callable[[str], None]] = None
    on_finding: Optional[Callable[[str, str], None]] = None
    on_replan: Optional[Callable[[str], None]] = None
    on_citation: Optional[Callable[[str, str, str], None]] = None


class GeminiClient:
    """Tiered Gemini client for RLM operations with timeout, retry, and rate limiting."""

    DEFAULT_TIMEOUT = 120.0  # 2 minutes
    MAX_RETRIES = 3
    DEFAULT_RPM = 60  # Requests per minute
    DEFAULT_BURST = 10  # Burst size

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        requests_per_minute: int = DEFAULT_RPM,
        burst_size: int = DEFAULT_BURST,
        cache: Optional["ResponseCache"] = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required")

        self.client = genai.Client(api_key=self.api_key)
        self.timeout = timeout
        self._cache = cache
        self._usage: dict[ModelTier, UsageStats] = {t: UsageStats() for t in ModelTier}
        self._rate_limiter = RateLimiter(requests_per_minute, burst_size)

    def _get_config(self, tier: ModelTier, system_prompt: Optional[str] = None) -> types.GenerateContentConfig:
        """Get generation config for a tier with system instruction."""
        mc = MODEL_CONFIGS[tier]

        # Use provided system prompt or default for tier
        if system_prompt is None:
            if tier == ModelTier.PRO:
                system_prompt = SYSTEM_PROMPT_PRO
            elif tier == ModelTier.FLASH:
                system_prompt = SYSTEM_PROMPT_FLASH
            else:  # LITE
                system_prompt = SYSTEM_PROMPT_WORKER

        config = types.GenerateContentConfig(
            temperature=mc.temperature,
            max_output_tokens=mc.max_output_tokens,
            system_instruction=system_prompt,
        )
        if mc.thinking_level:
            config.thinking_config = types.ThinkingConfig(thinking_level=mc.thinking_level)
        return config

    async def complete(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.FLASH,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        timeout: Optional[float] = None,
        use_cache: bool = True,
    ) -> str:
        """Generate completion using specified tier with timeout.

        Args:
            prompt: The prompt to send to the model
            tier: Model tier to use (LITE, FLASH, PRO)
            system_prompt: Optional custom system prompt (uses tier default if None)
            tools: Optional tools for function calling
            timeout: Optional custom timeout
            use_cache: Whether to use response cache (default True)

        Returns:
            The model's response text
        """
        mc = MODEL_CONFIGS[tier]
        config = self._get_config(tier, system_prompt)  # Pass system_prompt to config
        # timeout=0 means no timeout, None uses default
        request_timeout = timeout if timeout is not None else self.timeout
        no_timeout = (request_timeout == 0)

        # Build cache key (only cache if no tools and cache enabled)
        cache_enabled = use_cache and self._cache and not tools
        # Include tier in cache key since different tiers have different system prompts
        cache_key_prompt = f"{tier.value}:{prompt}"

        # Check cache first
        if cache_enabled:
            cached = self._cache.get(cache_key_prompt, mc.model_id)
            if cached:
                logger.debug(f"Cache hit for {mc.model_id}")
                return cached

        if tools:
            config.tools = tools

        # System prompt is now in config.system_instruction, just send user content
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )
        ]

        logger.debug(f"Calling {mc.model_id} with {len(prompt)} chars")

        # Acquire rate limit token
        await self._rate_limiter.acquire()

        # Try primary model with retry, fallback on 503
        model_to_use = mc.model_id
        max_attempts = 2
        last_error = None

        for attempt in range(max_attempts):
            try:
                api_call = asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_to_use,
                    contents=contents,
                    config=config,
                )
                if no_timeout:
                    response = await api_call
                else:
                    response = await asyncio.wait_for(api_call, timeout=request_timeout)
                break  # Success
            except asyncio.TimeoutError:
                logger.error(f"API call to {model_to_use} timed out after {request_timeout}s")
                raise TimeoutError(f"API call timed out after {request_timeout}s")
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Check for 503 overloaded error
                if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str.lower():
                    if attempt < max_attempts - 1:
                        logger.warning(f"{model_to_use} overloaded, retrying in 2s... (attempt {attempt + 1}/{max_attempts})")
                        await asyncio.sleep(2)
                    elif mc.fallback_model_id:
                        # Switch to fallback model
                        logger.warning(f"{model_to_use} still overloaded after {max_attempts} attempts, falling back to {mc.fallback_model_id}")
                        model_to_use = mc.fallback_model_id
                        # One more attempt with fallback
                        try:
                            fallback_call = asyncio.to_thread(
                                self.client.models.generate_content,
                                model=model_to_use,
                                contents=contents,
                                config=config,
                            )
                            if no_timeout:
                                response = await fallback_call
                            else:
                                response = await asyncio.wait_for(fallback_call, timeout=request_timeout)
                            break  # Fallback succeeded
                        except Exception as fallback_error:
                            raise fallback_error
                    else:
                        raise e
                else:
                    raise e
        else:
            # All retries exhausted
            if last_error:
                raise last_error

        # Track usage
        self._usage[tier].requests += 1
        # Estimate tokens (actual count would require response metadata)
        estimated_input = len(prompt) // 4
        estimated_output = len(response.text) // 4 if response.text else 0
        self._usage[tier].add(estimated_input, estimated_output)

        response_text = response.text or ""

        # Store in cache
        if cache_enabled and response_text:
            self._cache.set(cache_key_prompt, mc.model_id, response_text)
            logger.debug(f"Cached response for {mc.model_id}")

        logger.debug(f"Got response: {len(response_text)} chars")
        return response_text

    async def complete_with_retry(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.FLASH,
        system_prompt: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
    ) -> str:
        """Complete with exponential backoff retry."""
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.complete(prompt, tier, system_prompt)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)

        logger.error(f"All {max_retries} attempts failed")
        raise last_error

    async def complete_with_history(
        self,
        messages: list[dict],
        tier: ModelTier = ModelTier.FLASH,
    ) -> str:
        """Generate completion with conversation history."""
        mc = MODEL_CONFIGS[tier]
        config = self._get_config(tier)

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])]
            ))

        # Acquire rate limit token
        await self._rate_limiter.acquire()

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.models.generate_content,
                    model=mc.model_id,
                    contents=contents,
                    config=config,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"API call timed out after {self.timeout}s")

        self._usage[tier].requests += 1
        return response.text

    async def batch_complete(
        self,
        prompts: list[str],
        tier: ModelTier = ModelTier.LITE,
        max_concurrent: int = 5,
    ) -> list[str]:
        """Process multiple prompts in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(prompt: str) -> str:
            async with semaphore:
                return await self.complete(prompt, tier=tier)

        tasks = [process_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def get_usage(self) -> dict[str, UsageStats]:
        """Get usage statistics per tier."""
        return {tier.value: stats for tier, stats in self._usage.items()}

    def get_total_cost(self) -> float:
        """Get total estimated cost across all tiers."""
        total = 0.0
        for tier, stats in self._usage.items():
            mc = MODEL_CONFIGS[tier]
            total += stats.input_tokens * mc.cost_per_1m_input / 1_000_000
            total += stats.output_tokens * mc.cost_per_1m_output / 1_000_000
        return total

    def reset_usage(self):
        """Reset usage counters."""
        self._usage = {t: UsageStats() for t in ModelTier}
