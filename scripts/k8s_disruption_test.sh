#!/usr/bin/env bash
# Closed-loop traffic from inside the cluster while API pods are disrupted.
#
#   scripts/k8s_disruption_test.sh rollout   # kubectl rollout restart (normal deploy)
#   scripts/k8s_disruption_test.sh delete    # graceful pod deletion (node drain, eviction)
#   scripts/k8s_disruption_test.sh kill      # --grace-period=0 --force (kubelet still sends SIGTERM)
#   scripts/k8s_disruption_test.sh crash     # SIGKILL via the container runtime (expect errors)
#
# Requires the testing overlay (deploy/k8s-testing) on the kind cluster from `make k8s-up`.
set -euo pipefail

MODE=${1:-rollout}
CONTEXT=${CONTEXT:-kind-lumina}
REQUESTS=${REQUESTS:-1500}
k() { kubectl --context "$CONTEXT" -n lumina "$@"; }

k delete pod loadgen --ignore-not-found >/dev/null
k run loadgen --restart=Never --image=lumina-sql-agent:local --image-pull-policy=IfNotPresent \
  --overrides='{"spec":{"securityContext":{"runAsUser":0}}}' --command -- \
  sh -c "python scripts/load_test.py --base-url http://lumina-api:8000 --mock-url http://mock-llm:9000 \
         --levels 10 --repeats 1 --min-requests $REQUESTS --label k8s-$MODE >/dev/null && cat eval/results/load-k8s-*.json" \
  >/dev/null
k wait --for=condition=Ready pod/loadgen --timeout=60s >/dev/null
sleep 15  # warm-up plus steady traffic before the disruption

case "$MODE" in
  rollout)
    k rollout restart deployment/lumina-api >/dev/null
    k rollout status deployment/lumina-api --timeout=300s >/dev/null
    ;;
  delete)
    k delete pod "$(k get pods -l app.kubernetes.io/name=lumina-api -o name | head -1 | cut -d/ -f2)" >/dev/null
    ;;
  kill)
    # Force deletion still lets the kubelet send SIGTERM, so in-flight requests can drain.
    k delete pod "$(k get pods -l app.kubernetes.io/name=lumina-api -o name | head -1 | cut -d/ -f2)" \
      --grace-period=0 --force >/dev/null 2>&1
    ;;
  crash)
    # SIGKILL the API process through the container runtime: no drain, in-flight requests are lost.
    NODE=${KIND_NODE:-lumina-control-plane}
    docker exec "$NODE" sh -c 'crictl stop --timeout 0 "$(crictl ps --name "^api$" -q | head -1)"' >/dev/null
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

k wait --for=jsonpath='{.status.phase}'=Succeeded pod/loadgen --timeout=600s >/dev/null
k logs loadgen | python3 -c '
import json, sys
level = json.load(sys.stdin)["levels"][0]
print(json.dumps({key: level["raw_trials"][0][key] for key in
      ("requests", "status_counts", "error_rate", "agent_success_rate", "throughput_ok_rps", "latency_ok_ms")}))'
k delete pod loadgen >/dev/null
