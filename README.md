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


## Prerequisites

- pdfium: for reading pdf files and processing it
    - [pdfium dylib binaries](https://github.com/bblanchon/pdfium-binaries/releases)
    - Go to the above link
    - As we are using this as a dylib, as long as you don't use the `create_gif` function, you don't have to install this

```bash
wget <link>
mkdir pdfium
tar -xvzf <what you downloaded.tgz> -C pdfium
```


## Running tests

```bash
just test
```
