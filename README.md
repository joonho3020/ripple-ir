# Ripple-IR: Hardware intermediate representation

Graph based intermediate representation for RTL design.

## Goals

- Ergonomics first
    - We want to make complex pass writing viable and robust
    - Must be able to traverse the graph in both high-level form with circuit semantics as well as low-level form to analyze/modify individual circuit elememnts
- Clear separation of structural vs behavioral form. We want to preserve the behavioral semantics of the circuit and pass it directly to synthesis to achieve higher QoR
- Efficient subcircuit identification
    - This enables aggressive circuit deduplication (copy-on-write during pass exection to save memory)
    - Fine-grained incrementalism

## Running tests

```bash
just test
```
