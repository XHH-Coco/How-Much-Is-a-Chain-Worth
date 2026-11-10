.PHONY: check test env-verify fingerprint clean

check:            ## repository self-check, no third-party packages needed
	python scripts/check_repo.py

test:             ## unit tests, stdlib only
	python -m unittest discover -s tests -t . -v

env-verify:       ## compare the three conda envs against environment/pins.json
	python environment/verify_env.py

fingerprint:      ## write one env fingerprint per run (called by every runner)
	python -m b2.cli fingerprint --out results/env_fingerprints/manual.json

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
