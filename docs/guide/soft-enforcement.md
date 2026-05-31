[← Back to README](../../README.md)

# Soft Enforcement

## Built-in: `finding="not-found"` auto-skip

When the agent reports `finding="not-found"`, search enforcement is automatically skipped — there is no evidence to match by definition. No configuration needed.

## Custom skip rules

For other cases where certain record types should bypass search enforcement (e.g. a "kb" source type comes from a knowledge base rather than a search), use `skip_enforcement` on `record_tool()`.

### skip_enforcement (dict form)

Skip when any field matches a listed value:

```python
record = fence.record_tool(
    extra_fields=["finding", "source_type"],
    skip_enforcement={"source_type": ["kb", "web", "derived"]},
)
```

### skip_enforcement (callable form)

Full flexibility:

```python
record = fence.record_tool(
    skip_enforcement=lambda kw: kw.get("confidence", 0) > 0.9,
)
```

Document enforcement (`claim_in_document` vs `set_document()`) is always checked regardless of skip.
