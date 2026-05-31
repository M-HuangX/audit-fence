[← Back to README](../../README.md)

# Soft Enforcement

Sometimes certain record types should bypass search enforcement — a "not-found" finding doesn't have evidence to match, a "kb" source type comes from a knowledge base rather than a search. audit-fence supports conditional enforcement via `skip_enforcement` on `record_tool()`.

### skip_enforcement (dict form)

Skip when any field matches a listed value:

```python
record = fence.record_tool(
    extra_fields=["finding"],
    skip_enforcement={"finding": ["not-found", "derived"]},
)

# Multiple fields — skip if ANY matches
record = fence.record_tool(
    extra_fields=["finding", "source_type"],
    skip_enforcement={
        "finding": ["not-found"],
        "source_type": ["kb", "web"],
    },
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
