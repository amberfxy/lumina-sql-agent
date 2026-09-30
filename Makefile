IMAGE ?= lumina-sql-agent:local
KIND_CLUSTER ?= lumina
KIND ?= kind
CONCURRENCY ?= 8

.PHONY: up up-all down logs test lint eval validate-gold k8s-up k8s-down k8s-status

up:  ## API, UI, Postgres (seeded), Redis
	docker compose up -d --build

up-all:  ## Everything, including Prometheus, Grafana, and DynamoDB Local
	docker compose --profile observability --profile dynamodb up -d --build

down:
	docker compose --profile observability --profile dynamodb --profile test down

logs:
	docker compose logs -f api

test:  ## Unit + integration tests against the compose Postgres/Redis
	docker compose --profile test run --rm --build tests pytest -q -p no:cacheprovider

lint:
	docker compose --profile test run --rm --build tests sh -c "ruff check --no-cache . && ruff format --no-cache --check ."

validate-gold:
	docker compose run --rm api python -m evaluation.run_eval --validate-gold

eval:  ## Run the NL-to-SQL evaluation (needs an LLM key in .env)
	docker compose run --rm -v ./evaluation/results:/app/evaluation/results api \
		python -m evaluation.run_eval --concurrency $(CONCURRENCY)

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
