# Governor — reproducibility entry points.
# Everything here runs WITHOUT an API key except `collect` and `curve`.

PY ?= python

.PHONY: help test smoke verify gate governor report clean-sessions all keys

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

test:            ## full regression suite (no network, no API key)
	$(PY) -m pytest tests/ -q

smoke:           ## end-to-end smoke: both families, MCP, ActionExecutor, traps, ledger
	$(PY) scripts/e2e_smoke.py

verify:          ## re-verify every recorded experiment from disk
	@$(PY) -c "from governor.harness.ledger import index; \
	rows=index(); \
	[print(f\"{r['exp_id']:<24} {r['verdict']:<30} verifies={r['verifies']}\") for r in rows]; \
	bad=[r['exp_id'] for r in rows if not r['verifies'] and r['verdict']!='UNFINALIZED']; \
	print('ALL FINALIZED EXPERIMENTS VERIFY' if not bad else f'FAILED: {bad}'); \
	raise SystemExit(1 if bad else 0)"

gate:            ## Phase 4R held-out ceiling gate (needs 24 evaluation items)
	$(PY) scripts/p4r_ceiling.py --boot 400

governor:        ## Phase 4R Governor test — refuses unless the gate passed
	$(PY) scripts/p4r_governor.py

mcp:             ## MCP stdio smoke over real JSON-RPC
	./scripts/mcp_smoke.sh

collect:         ## fill evaluation items as Groq quota refills (needs Groq key)
	$(PY) scripts/p4r_patient_collect.py

curve:           ## local Qwen backend curve (MLX, no network after download)
	$(PY) scripts/qwen_local_curve.py --n 8

all: test smoke verify   ## everything that needs no API key

keys:  ## show which provider API keys the harness can see
	@python -m governor.config
