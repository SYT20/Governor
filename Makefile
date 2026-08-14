PY ?= python3

.PHONY: test smoke run clean check

test:            ## invariant tests (no deps, ~40ms)
	$(PY) -m unittest discover -s tests -t . -v

smoke:           ## fast end-to-end sanity sweep
	$(PY) scripts/run_experiment.py --episodes 25 --warmup 60

run:             ## full sweep, writes results/governor.db
	$(PY) scripts/run_experiment.py --episodes 120 --warmup 120

check: test smoke ## what CI runs

clean:
	rm -rf results/*.db **/__pycache__ .pytest_cache
