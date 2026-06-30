"""Quick single-query test."""
import sys, os, asyncio, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Require API key from environment - never hardcode keys
if not os.environ.get('GEMINI_API_KEY'):
    print("ERROR: GEMINI_API_KEY environment variable required")
    sys.exit(1)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from irys.core.models import GeminiClient
from irys.rlm.engine import RLMEngine, RLMConfig
from irys.rlm.state import classify_query

REPO = r'C:\Users\devan\Downloads\CITIOM v Gulfstream\documents'

parse_errors = 0

def on_step(step):
    global parse_errors
    if 'JSON parse' in step.content:
        parse_errors += 1
        print(f'  [JSON ERROR] {step.content[:60]}...', flush=True)
    elif step.step_type.value == 'thinking':
        print(f'  {step.display[:80]}', flush=True)

async def main():
    global parse_errors
    query = 'What is the serial number of the aircraft?'
    print(f'Query: {query}', flush=True)
    print(f'Type: {classify_query(query)["type"]}', flush=True)
    print('Starting investigation...', flush=True)

    client = GeminiClient(api_key=os.environ['GEMINI_API_KEY'])
    engine = RLMEngine(client, RLMConfig(max_iterations=5, min_depth=1), on_step=on_step)

    start = time.time()
    state = await engine.investigate(query, REPO)
    elapsed = time.time() - start

    print(f'\n=== RESULTS ===', flush=True)
    print(f'Iterations: {len(state.facts_per_iteration)}', flush=True)
    print(f'Facts/iter: {state.facts_per_iteration}', flush=True)
    print(f'Total facts: {len(state.findings.get("accumulated_facts", []))}', flush=True)
    print(f'Citations: {len(state.citations)}', flush=True)
    print(f'Time: {elapsed:.1f}s', flush=True)
    print(f'JSON parse errors: {parse_errors}', flush=True)

if __name__ == '__main__':
    asyncio.run(main())
