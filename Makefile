IMAGE ?= lumina-sql-agent:local
KIND_CLUSTER ?= lumina
KIND ?= kind
# kustomize directory: deploy/k8s (real LLM key) or deploy/k8s-testing (in-cluster mock LLM)
K8S_DIR ?= deploy/k8s
KUBECTL = kubectl --context kind-$(KIND_CLUSTER)
CONCURRENCY ?= 8

.PHONY: up up-all down logs test lint eval eval-single-shot eval-guard eval-scripted validate-gold \
	loadtest-up loadtest bench-cache resilience k8s-up k8s-down k8s-status k8s-disruption

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

# Clients run in a separate, unconstrained container so they never compete with the 1-CPU API.
CLIENT_RUN = docker run --rm --network lumina_default -v "$(CURDIR)/eval/results:/app/eval/results" lumina-sql-agent:local

loadtest:  ## Concurrency sweep against the loadtest API (3 trials per level, median reported)
	$(CLIENT_RUN) python scripts/load_test.py --base-url http://api-loadtest:8000 --levels 10 25 50 100 150 200 300 400

bench-cache:  ## Cache hit vs miss latency against the loadtest API
	$(CLIENT_RUN) python scripts/bench_cache.py --base-url http://api-loadtest:8000

resilience:  ## Fault injection under traffic (LLM 429/500/timeout/malformed, Postgres down, Redis down)
	python3 scripts/resilience.py

k8s-up:  ## Local Kubernetes deployment on kind
	@$(KIND) get clusters | grep -qx $(KIND_CLUSTER) || $(KIND) create cluster --name $(KIND_CLUSTER)
	docker build --target base -t $(IMAGE) .
	$(KIND) load docker-image $(IMAGE) --name $(KIND_CLUSTER)
	$(KUBECTL) apply -f deploy/k8s/namespace.yaml
	$(KUBECTL) -n lumina create configmap postgres-init --from-file=db/init --dry-run=client -o yaml | $(KUBECTL) apply -f -
	$(KUBECTL) apply -k $(K8S_DIR)
	$(KUBECTL) -n lumina rollout status statefulset/postgres --timeout=180s
	$(KUBECTL) -n lumina rollout status deployment/redis --timeout=120s
	$(KUBECTL) -n lumina rollout status deployment/lumina-api --timeout=180s

k8s-status:
	$(KUBECTL) -n lumina get pods,svc,hpa,pdb,networkpolicy

k8s-disruption:  ## Rollout/delete/kill/crash under traffic (needs `make k8s-up K8S_DIR=deploy/k8s-testing`)
	for mode in rollout delete kill crash; do echo "== $$mode"; scripts/k8s_disruption_test.sh $$mode; \
		$(KUBECTL) -n lumina rollout status deployment/lumina-api --timeout=180s >/dev/null; done

k8s-down:
	$(KIND) delete cluster --name $(KIND_CLUSTER)
