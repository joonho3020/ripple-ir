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


## Uncompress some example input FIRRTL files

```bash
just uncompress
```

## Running tests

```bash
just test
```

## Running specific tests

Run this to see the list of tests:

```bash
just list
```

Run this to run a specific test:

```bash
just test_only <path of the test>
```

## Some ideas on incremental compilation support and graph-ISO

### Background

The benefit of incremental compilation is very obvious: the faster the compile times, the faster you can iterate on your design.
In the software world, incremental compilation has come a long way: the edit-run-debug loop has gotten so tight that things feel almost instant.
Yet, hardware compilers have failed to innovate in areas that enables agile hardware development.

So what makes incremental compilation for hardware different from software compilers?
First of all, in software there are well defined interfaces (e.g., function signatures) that can be used as incremental boundaries.
Conversely in hardware, this is not the case.
Although hardware module boundaries can be thought of as functions in software, the variance between module sizes are larger than functions.
For instance, if you make a change in a CPU module that is not contained in any other child module, then you would have to recompile the entire CPU which is very wasteful.

Furthermore, hardware compilers tends to flatten the module hierarchy to simplify the compiler design.
After the circuit is flattened, there are no module boundaries to rely on anymore.
To make things worse, synthesis tools performs optimizations that changes the physical circuit representation from the specification such that the functionality is preserved.
One example of this optimization is retiming: registers are moved around to meet timing constraints while preserving the design functionality.
These optimizations makes it difficult to identify correspondence points between the compiled output and the input circuit representation.

### Problem & high level idea

The above issues points us to a common problem: given two circuits represented in a graph form (circuits are graphs after all), how do we identify a common subgraph between the two and isolate different regions?
Well, the maximul common subgraph problem is a NP hard problem just like most problems we encountered during class!
However, I believe we can solve this for most cases that we run into during the hardware development cycle.
Also, we don't even have to search for the maximul subgraph: a large enough common subgraph is good enough.
Finally, it is perfectly okay if we run into a bad case where the complexity explodes: we can just compile the design from scratch in this case.

Seems like there are some prior work that seems relevant to what we are trying to do.
In [An Approximate Maximum Common Subgraph Algorithm for Large Digital Circuits](https://ieeexplore.ieee.org/abstract/document/5615521), it finds seemingly similar nodes.
Starting from these similarish looking nodes, it iteratively finds nodes that are similar between the two graphs.

The first step would be to replicate this work in our infrastructure.
Additionional details are explained in the [implementation section](#implementation).


### Plan

#### Obtaining background information

- [ ] Read [An Approximate Maximum Common Subgraph Algorithm for Large Digital Circuits](https://ieeexplore.ieee.org/abstract/document/5615521)
- [ ] Read [Graph hashing](https://arxiv.org/pdf/2002.06653)

#### Implementation

- Generate a bunch of chisel examples that are slightly different from each other
    - [Chisel examples](https://github.com/joonho3020/chisel-examples): we can generate FIRRTL files using this repo
    - [ ] Create a bunch of Chisel circuits that contains small diffs
        - ALU, Queue, Register file, pointer chasing, Small cache, GCD, FIR filter
    - We can move on to larger circuits after a while
- This repo can currently generate circuit graphs from the FIRRTL/CHIRRTL IR
    - It parses the textual format of CHIRRTL and converts it into a internal graph format
    - You can extend this repo to investigate various approaches described below
- Perform the diff and identify subgraphs that are isomorphic between different circuit versions
    - [ ] Reproduce this paper's algorithm: [An Approximate Maximum Common Subgraph Algorithm for Large Digital Circuits](https://ieeexplore.ieee.org/abstract/document/5615521)
    - [ ] (Experimental/researchy) See if we can augment the above algorithm by using merkel-tree type hashing

### Related work

- [An Approximate Maximum Common Subgraph Algorithm for Large Digital Circuits](https://ieeexplore.ieee.org/abstract/document/5615521)
- [Graph hashing](https://arxiv.org/pdf/2002.06653)
- [DAGs and Equivalence Classes of DAGs](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume18/acid03a-html/node2.html)

### Existing graph-iso libraries

- [implementation of the graph-hasing paper](https://github.com/calebh/dihash)
- [nauty-traces-rust](https://crates.io/crates/nauty-Traces-sys)
- [nautry-pet](https://docs.rs/nauty-pet/latest/nauty_pet/)
- VF2 algorithm
    - [VF2 in petgraph](https://docs.rs/petgraph/latest/petgraph/algo/isomorphism/index.html)
    - [Rust VF2 standalone](https://docs.rs/vf2/latest/vf2/)
