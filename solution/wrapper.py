"""YOUR mitigation + observability layer. The simulator calls mitigate() around the
opaque agent (a REAL LLM) for every request. This is the ONLY place observability can
live -- the agent is silent. Legal moves: retry / cache / route / guardrail / sanitize
/ fallback / session-reset / PROMPT ROUTING, plus your own logging/tracing/metrics.
Illegal: hardcoding answers, importing the agent internals, reading instructor files,
network exfiltration.

  call_next(question, config) -> result   # the only way to reach the black box
  context = {"session_id","turn_index","qid","cache": <shared dict>, "cache_lock": <Lock>}
  result  = {"answer","status","steps","trace","meta":{latency_ms,usage,...}}

PROMPT ROUTING: you can override the agent's system prompt PER REQUEST by setting it in
the config you pass to call_next, e.g.:
    conf = dict(config); conf["system_prompt"] = my_better_prompt
    result = call_next(question, conf)
(Or just edit solution/prompt.txt for a single static prompt used on every request.)
"""
from __future__ import annotations

import time
import re
from telemetry.logger import logger
from telemetry.cost import cost_from_usage

def mitigate(call_next, question, config, context):
    t0 = time.time()
    
    # Sanitize input: strip out injected notes that try to bypass prices
    if isinstance(question, str):
        if "GHI CHÚ:" in question:
            question = re.sub(r'GHI CHÚ:.*$', '', question, flags=re.MULTILINE|re.DOTALL)
        if "Ghi chú:" in question:
            question = re.sub(r'Ghi chú:.*$', '', question, flags=re.MULTILINE|re.DOTALL)
            
    try:
        result = call_next(question, config)
        if not isinstance(result, dict):
            result = {"answer": None, "status": "unknown_error", "steps": 0, "trace": [], "meta": {}}
    except Exception as e:
        print(f"\n[LỖI API HOẶC AGENT]: {e}\n")
        return {"answer": None, "status": "api_error", "steps": 0, "trace": [], "meta": {}}
        
    try:
        meta = result.get("meta") or {}
        latency_ms = meta.get("latency_ms")
        if latency_ms is None:
            latency_ms = int((time.time() - t0) * 1000)
            
        usage = meta.get("usage") or {}
        model = meta.get("model") or "gpt-5.4-nano"
        
        # Calculate approximate cost
        cost = cost_from_usage(model, usage)
        
        tools = meta.get("tools_used") or []
        
        # Log observability data
        logger.log_event("CALL", {
            "qid": context.get("qid"),
            "session_id": context.get("session_id"),
            "turn_index": context.get("turn_index"),
            "latency_ms": latency_ms,
            "cost": cost,
            "tools_used": len(tools),
            "steps": result.get("steps") or 0,
            "status": result.get("status")
        })
    except Exception as e:
        # Prevent wrapper errors from failing the request
        print(f"Observability error: {e}")
        
    return result
