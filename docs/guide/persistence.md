[← Back to README](../../README.md)

# Persistence

Fence state can be serialized for audit trails that survive process restarts.

### Single fence

```python
import json

state = fence.snapshot()
with open("fence_state.json", "w") as f:
    json.dump(state, f)

restored = Fence.restore(json.load(open("fence_state.json")))
```

### FenceGroup — preserves links

```python
state = group.snapshot()
with open("group_state.json", "w") as f:
    json.dump(state, f)

restored_group = FenceGroup.restore(json.load(open("group_state.json")))
```

The snapshot captures search history, rejections, configuration, and link topology.
