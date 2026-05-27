#!/bin/sh
set -u

managed_by="pi-llama-cpp-provider"
root=${1:-${LLAMA_CPP_DIR:-}}
[ -n "$root" ] || exit 0

client_dir=${LLAMA_CPP_CLIENT_DIR:-$root/clients}
state_file=${LLAMA_CPP_STATE_FILE:-$root/server.json}
log_file=${LLAMA_CPP_LOG_FILE:-$root/log}
lease_ttl_s=${LLAMA_CPP_LEASE_TTL_S:-45}
poll_s=${LLAMA_CPP_WATCHDOG_POLL_S:-2}
shutdown_grace_s=${LLAMA_CPP_SHUTDOWN_GRACE_S:-60}

log() { mkdir -p "$root" 2>/dev/null || true; printf '[%s] llama-cpp-watchdog: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$log_file" 2>/dev/null || true; }
pid_alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
mtime_sec() {
  mt=$(stat -c %Y "$1" 2>/dev/null | head -1 || true)
  case "$mt" in ''|*[!0-9]*) mt=$(stat -f %m "$1" 2>/dev/null | head -1 || true);; esac
  case "$mt" in ''|*[!0-9]*) echo 0;; *) echo "$mt";; esac
}
process_args() { ps -p "$1" -o args= 2>/dev/null || true; }
process_start() { ps -p "$1" -o lstart= 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || true; }
json_string_field() { sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$2" 2>/dev/null | head -1; }
looks_like_server() { process_args "$1" | grep -Eq '(^|[/[:space:]])llama-server([[:space:]]|$)'; }
state_pid() { sed -n 's/.*"pid"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$state_file" 2>/dev/null | head -1; }
state_port() { sed -n 's/.*"port"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$state_file" 2>/dev/null | head -1; }

active_lease_count() {
  mkdir -p "$client_dir" 2>/dev/null || true
  count=0; now=$(date +%s)
  for file in "$client_dir"/*.json; do
    [ -e "$file" ] || continue
    name=${file##*/}; pid=${name%.json}; stale=0
    grep -q '"managedBy"[[:space:]]*:[[:space:]]*"pi-llama-cpp-provider"' "$file" 2>/dev/null || stale=1
    grep -q '"usesLlamaCpp"[[:space:]]*:[[:space:]]*true' "$file" 2>/dev/null || stale=1
    pid_alive "$pid" || stale=1
    lease_start=$(json_string_field processStart "$file"); proc_start=$(process_start "$pid")
    [ -n "$lease_start" ] && [ -n "$proc_start" ] && [ "$lease_start" = "$proc_start" ] || stale=1
    mt=$(mtime_sec "$file"); [ $((now - mt)) -le "$lease_ttl_s" ] || stale=1
    if [ "$stale" -eq 1 ]; then rm -f "$file" 2>/dev/null || true; else count=$((count + 1)); fi
  done
  echo "$count"
}

managed_server_pid() { pid=$(state_pid); [ -n "$pid" ] && pid_alive "$pid" && looks_like_server "$pid" && echo "$pid"; }
server_has_clients() {
  pid=$(managed_server_pid || true); [ -n "$pid" ] || return 1
  command -v lsof >/dev/null 2>&1 || return 1
  lsof -nP -a -p "$pid" -iTCP -sTCP:ESTABLISHED 2>/dev/null | awk 'NR > 1 { f = 1 } END { exit f ? 0 : 1 }'
}
stop_server() {
  pid=$(managed_server_pid || true)
  [ -n "$pid" ] || { rm -f "$state_file" 2>/dev/null || true; return 0; }
  log "stopping llama-server pid=$pid"
  kill -TERM "$pid" 2>/dev/null || true
  waited=0
  while pid_alive "$pid" && [ "$waited" -lt "$shutdown_grace_s" ]; do sleep 1; waited=$((waited + 1)); done
  if pid_alive "$pid"; then log "sending SIGKILL to pid=$pid"; kill -KILL "$pid" 2>/dev/null || true; sleep 1; fi
  pid_alive "$pid" || rm -f "$state_file" 2>/dev/null || true
}

log "started for $root"
waiting=0
while :; do
  if [ "$(active_lease_count)" -eq 0 ]; then
    if server_has_clients; then
      [ "$waiting" -eq 0 ] && log "no active leases, but server still has clients; waiting"
      waiting=1; sleep "$poll_s"; continue
    fi
    log "no active leases; stopping server"
    stop_server
    log "exiting"
    exit 0
  fi
  waiting=0; sleep "$poll_s"
done
