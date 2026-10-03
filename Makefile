IMAGE ?= lumina-sql-agent:local
KIND_CLUSTER ?= lumina
KIND ?= kind
CONCURRENCY ?= 8

.PHONY: up up-all down logs test lint eval eval-single-shot eval-guard eval-scripted validate-gold \
	loadtest-up loadtest bench-cache k8s-up k8s-down k8s-status

up:  ## API, UI, Postgres (seeded), Redis
	docker compose up -d --build

up-all:  ## Everything, including Prometheus, Grafana, and DynamoDB Local
	docker compose --profile observability --profile dynamodb up -d --build

down:
	docker compose --profile observability --profile dynamodb --profile test --profile loadtest down

logs:
	docker compose logs -f api

test:  ## Unit + integration tests against the compose Postgres/Redis
	docker compose --profile test run --rm --build tests pytest -q -p no:cacheprovider

lint:
	docker compose --profile test run --rm --build tests sh -c "ruff check --no-cache . && ruff format --no-cache --check ."

EVAL_RUN = docker compose run --rm -v ./eval/results:/app/eval/results api python scripts/run_eval.py

validate-gold:  ## Every gold query runs and returns rows
	$(EVAL_RUN) validate-gold

eval-guard:  ## SQL guard vs adversarial corpus + gold false positives (no LLM)
	$(EVAL_RUN) guard

eval-scripted:  ## Deterministic failure-mode suite against Postgres (no LLM)
	$(EVAL_RUN) scripted

eval:  ## Real-model evaluation on the categorized dataset (needs an LLM key in .env)
	$(EVAL_RUN) model --concurrency $(CONCURRENCY)

eval-single-shot:  ## Same, with self-correction disabled (for the self-correction ablation)
	$(EVAL_RUN) model --concurrency $(CONCURRENCY) --max-attempts 1

loadtest-up:  ## Mock LLM + a second API instance on :8002 for load/cache/resilience tests
	docker compose --profile loadtest up -d --build mock-llm api-loadtest

loadtest:  ## Concurrency sweep 10/25/50/100 against the loadtest API
	docker compose --profile loadtest run --rm --no-deps -v ./eval/results:/app/eval/results api-loadtest \
		python scripts/load_test.py --base-url http://api-loadtest:8000

bench-cache:  ## Cache hit vs miss latency against the loadtest API
	docker compose --profile loadtest run --rm --no-deps -v ./eval/results:/app/eval/results api-loadtest \
		python scripts/bench_cache.py --base-url http://api-loadtest:8000

k8s-up:  ## Local Kubernetes deployment on kind
	@$(KIND) get clusters | grep -qx $(KIND_CLUSTER) || $(KIND) create cluster --name $(KIND_CLUSTER)
	docker build --target base -t $(IMAGE) .
	$(KIND) load docker-image $(IMAGE) --name $(KIND_CLUSTER)
	kubectl apply -f deploy/k8s/namespace.yaml
	kubectl -n lumina create configmap postgres-init --from-file=db/init --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -k deploy/k8s
	kubectl -n lumina rollout status statefulset/postgres --timeout=180s
	kubectl -n lumina rollout status deployment/redis --timeout=120s
	kubectl -n lumina rollout status deployment/lumina-api --timeout=180s

k8s-status:
	kubectl -n lumina get pods,svc,hpa,pdb

k8s-down:
	$(KIND) delete cluster --name $(KIND_CLUSTER)
