# Building an RTL Simulation Backend

## Goals

Existing simulators such as Verilator suffers from long compilation times.
This is because under the hood, it generates a giant blob of C++ code that gets passed on to gcc which chokes on large designs.
There is no fundamental reason that prevents us from generating [Cranelift](https://cranelift.dev) IR directly from the circuit representation.
In addition to decreasing the compilation time, we can further work on performance optimizations such as deduplication and vectorization in the future.


## Steps

### 1. Get familiar with Rust

Create a fresh repo and see if you can create a small graph using the [petgraph](https://docs.rs/petgraph/latest/petgraph/) library.
Check if you can perform BFS using this library (don't use the existing BFS APIs).

### 2. High level concepts of [this](https://github.com/joonho3020/ripple-ir) repo

This repo is a compiler infrastructure for operating on the outputs of [Chisel](https://www.chisel-lang.org) based designs.
Although there already exists such infrastructure called [CIRCT](https://circt.llvm.org), having to work with MLIR can be daunting.
We try to create a cleaner intermediate representation so that it is easier to perform circuit analysis and transformations.

The high level overview of what happens is this:

- We elaborate Chisel and obtain a textual representation of CHIRRTL (some examples can be found in the `test-inputs` directory). This happens prior to invoking any code in this repo
- We parse the CHIRRTL file into an AST
- We convert the AST into our internal IR called FIR
- We can now perform analysis and transformations on the FIR format

### 3. Setup repo and run some code

Setup instructions for the repo is in [installing dependencies](https://github.com/joonho3020/ripple-ir?tab=readme-ov-file#prerequisites).

Now let's get started by running some code.

First uncompress some example input files (the `test-inputs` directory contains example CHIRRTL files):

```bash
just uncompress
```

Next, let's convert the CHIRRTL file into our FIR format, and convert it back again to CHIRRTL by running [main.rs](https://github.com/joonho3020/ripple-ir/blob/main/src/main.rs).

```bash
cargo run -- --input <path to input firrtl file> --output <path to output firrtl file> --firrtl-version chirrtl

# For Example
cargo run -- --input test-inputs/Adder.fir --output test-outputs/HELLO.fir --firrtl-version chirrtl
```


### 4. Understand the FIR format by reading the code

FIR is simply a graph representation of a circuit.
Nodes represent circuit elements such as registers, memory, wires, combinational operators (add, sub, etc) and muxes.
Edges represents connections between the elements.

To visualize the FIR format, you can call the `.export` function on `FirIR` instance.
This will produce a PDF containing the visualized version of the FIR format.

The directory is organized like the following:

```
docs/     : contains some documentation about this project
src/
  common/ : contains some utility code
  ir/     : contains the IR representations and additional datastructures for bookkeeping
  passes/ : compiler passes on each representation. Can ignore the RIR format for now
```


### 5. Convert the FIR schema to an IR more suitable for building RTL simulations

The FIR form has edges that represent aggregate wires.
Aggregate wires contain multiple signals and signals can flow in different directions as well.
Also, the module hierarchy is currently preserved in FIR.
In order to make the graph traversal easy, we would want to create separate edges for separate wires (i.e., flatten aggregate types) as well as flatten the module hierarchy.
You can do this here [rusty-rtlsim](https://github.com/joonho3020/rusty-rtlsim).

### 6. Understand how RTL simulators works by writing an interpreted version


Take a look at the following links and implement the interpretter using your IR format

- [How RTL simulation works](https://joonho3020.github.io/articles/rtl-simulation.html)
- [Python RTL simulator interpretter](https://github.com/evanjyli/chisel-examples/blob/main/simulator_2.py)

### 7. Hook this up to Cranelift

Some useful links:

- [codegen](https://docs.rs/cranelift-codegen/latest/cranelift_codegen/)
- [frontend](https://docs.rs/cranelift-frontend/latest/cranelift_frontend/)
- [module](https://docs.rs/cranelift-module/latest/cranelift_module/)
- [jit](https://docs.rs/cranelift-jit/latest/cranelift_jit/)

Example of doing this for some simple gates:

- [Cranelift simple combinational logic example](https://github.com/joonho3020/rusty-rtlsim/blob/main/src/main.rs)
