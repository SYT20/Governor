#!/usr/bin/env bash
# End-to-end MCP smoke test over real stdio JSON-RPC. No API key required.
set -euo pipefail
cd "$(dirname "$0")/.."
{
  echo '{"jsonrpc":"2.0","id":1,"method":"initialize"}'
  echo '{"jsonrpc":"2.0","method":"notifications/initialized"}'
  echo '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
  echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"gate_status","arguments":{}}}'
  echo '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"experiment_index","arguments":{}}}'
} | python -m governor.mcp.server
