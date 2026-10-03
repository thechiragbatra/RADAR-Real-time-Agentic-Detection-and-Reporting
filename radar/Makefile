.PHONY: setup simulate train serve test lint demo eval-cases eval-heuristic eval-bedrock eval-compare \
        docker-build loadtest tf-init tf-plan tf-apply tf-destroy bootstrap-aws

PY ?= python
DATA ?= data
MODELS ?= models
ALB ?= http://localhost:8000

setup:            ## install the package with dev extras
	pip install -e ".[dev]"

simulate:         ## generate 90 days of synthetic transactions (~600k rows, ~20s)
	$(PY) -m radar.simulator.generate --out $(DATA) --fraud-user-fraction 0.06

train:            ## train, evaluate and promote a model into $(MODELS)
	$(PY) -m radar.model.train --data $(DATA) --models $(MODELS) --test-days 14 --promote

serve:            ## run the scoring API locally on :8000
	RADAR_MODEL_URI=$(MODELS) $(PY) -m radar.scoring.api

test:             ## unit + integration tests (moto for AWS)
	pytest -q -p no:warnings

lint:
	ruff check src tests && ruff format --check src tests

demo:             ## stream 3000 transactions through the full path, investigate flagged ones
	$(PY) scripts/local_stream_demo.py --last 3000 --investigate 8

eval-cases:       ## build the 100-case labelled evaluation set
	$(PY) -m radar.evals.harness build-cases --n 100

eval-heuristic:   ## baseline investigator on the case set
	$(PY) -m radar.evals.harness run --backend heuristic

eval-bedrock:     ## LLM agent on the case set (needs AWS credentials + Bedrock model access)
	$(PY) -m radar.evals.harness run --backend bedrock

eval-compare:
	$(PY) -m radar.evals.harness compare

docker-build:
	docker build -f docker/Dockerfile.scorer -t radar-scorer .
	docker build -f docker/Dockerfile.lambda -t radar-lambda .

loadtest:         ## 50 users for 2 minutes against $(ALB)
	mkdir -p loadtest/results && locust -f loadtest/locustfile.py --host $(ALB) --headless -u 50 -r 10 -t 2m --csv loadtest/results/run

tf-init:
	cd infra/terraform && terraform init

tf-plan:
	cd infra/terraform && terraform plan -var-file=envs/dev.tfvars

tf-apply:
	cd infra/terraform && terraform apply -var-file=envs/dev.tfvars

tf-destroy:
	cd infra/terraform && terraform destroy -var-file=envs/dev.tfvars

bootstrap-aws:    ## upload models + reference data after tf-apply (BUCKET and TABLE from terraform output)
	$(PY) scripts/bootstrap_aws.py --bucket $(BUCKET) --table $(TABLE) --models $(MODELS) --data $(DATA)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'
