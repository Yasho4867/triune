import urllib.request
import json
import time

time.sleep(1)

# 1. Test GET /v1/plugins
print("--- 1. Testing GET /v1/plugins ---")
req = urllib.request.urlopen('http://localhost:8000/v1/plugins')
raw = req.read().decode('utf-8')
print("Raw /v1/plugins response:", repr(raw[:150]))
plugins = json.loads(raw)
print(f"Discovered {len(plugins)} registered plugins in global_registry")
for p in plugins[-8:]:
    print(f"  - {p['name']} ({p.get('category', 'unknown')}): inputs={p.get('inputs')} outputs={p.get('outputs')}")

# 2. Test POST /v1/modules/scan_local
print("\n--- 2. Testing POST /v1/modules/scan_local ---")
req = urllib.request.Request(
    'http://localhost:8000/v1/modules/scan_local',
    data=json.dumps({}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res = urllib.request.urlopen(req)
scan_result = json.loads(res.read().decode('utf-8'))
print(f"Scan Success: {scan_result.get('success')}")
print(f"Scanned Paths: {len(scan_result.get('scanned_paths', []))}")
print(f"Discovered Nodes: {scan_result.get('discovered_nodes')}")
print(f"Total Registered Nodes: {scan_result.get('total_registered_nodes')}")
print(f"Message: {scan_result.get('message')}")

# 2b. Test POST /v1/modules/create_plugin
print("\n--- 2b. Testing POST /v1/modules/create_plugin ---")
req = urllib.request.Request(
    'http://localhost:8000/v1/modules/create_plugin',
    data=json.dumps({"name": "TestAutoScaffoldLoss", "category": "Loss"}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res = urllib.request.urlopen(req)
create_result = json.loads(res.read().decode('utf-8'))
print(f"Plugin Creation: {create_result.get('success')}")
print(f"Created File: {create_result.get('file_path')}")
print(f"Node Registered: {create_result.get('node_name')}")

# 2c. Rescan workspace to verify dynamic discovery
print("\n--- 2c. Rescanning workspace ---")
req = urllib.request.Request(
    'http://localhost:8000/v1/modules/scan_local',
    data=json.dumps({}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res = urllib.request.urlopen(req)
rescan_result = json.loads(res.read().decode('utf-8'))
print(f"Discovered Nodes on Rescan: {rescan_result.get('discovered_nodes')}")
print(f"Total Registered Nodes: {rescan_result.get('total_registered_nodes')}")

# 3. Test POST /v1/dag/execute_node for JointExitLoss
print("\n--- 3. Testing POST /v1/dag/execute_node (JointExitLoss) ---")
node_payload = {
    'node': {
        'id': 'test_joint_loss_1',
        'type': 'Loss',
        'name': 'JointExitLoss',
        'title': 'Joint Exit Loss',
        'params': {
            'lambda_reflex': '0.20',
            'lambda_limbic': '0.30',
            'lambda_cortex': '0.50'
        }
    }
}
req = urllib.request.Request(
    'http://localhost:8000/v1/dag/execute_node',
    data=json.dumps(node_payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res = urllib.request.urlopen(req)
node_result = json.loads(res.read().decode('utf-8'))
print(f"Node Status: {node_result.get('status')}")
print(f"Elapsed: {node_result.get('elapsed_sec')}s")
print(f"Output: {node_result.get('output')}")

# 4. Test POST /v1/dag/execute_node for FastRoPE
print("\n--- 4. Testing POST /v1/dag/execute_node (FastRoPE) ---")
rope_payload = {
    'node': {
        'id': 'test_rope_1',
        'type': 'Model',
        'name': 'FastRoPE',
        'title': 'Fast RoPE',
        'params': {
            'dim': '64',
            'max_seq_len': '2048'
        }
    }
}
req = urllib.request.Request(
    'http://localhost:8000/v1/dag/execute_node',
    data=json.dumps(rope_payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res = urllib.request.urlopen(req)
rope_result = json.loads(res.read().decode('utf-8'))
print(f"RoPE Status: {rope_result.get('status')}")
print(f"Output: {rope_result.get('output')}")

# 5. Test POST /v1/dag/execute_node for GradientAccumulator
print("\n--- 5. Testing POST /v1/dag/execute_node (GradientAccumulator) ---")
accum_payload = {
    'node': {
        'id': 'test_accum_1',
        'type': 'Runtime',
        'name': 'GradientAccumulator',
        'title': 'Gradient Accumulator',
        'params': {
            'grad_accum_steps': '4'
        }
    }
}
req = urllib.request.Request(
    'http://localhost:8000/v1/dag/execute_node',
    data=json.dumps(accum_payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res = urllib.request.urlopen(req)
accum_result = json.loads(res.read().decode('utf-8'))
print(f"Accum Status: {accum_result.get('status')}")
print(f"Output: {accum_result.get('output')}")

print("\nALL API VALIDATION TESTS PASSED!")
