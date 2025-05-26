#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [Building New Abstractions for HDL IRs],
  abstract: [
    HDL compilers enables agile hardware development by design reuse.
    However, existing compilers are built on intermediate representation(IR)s that have several deficiencies.
    We propose and implement a new HDL IR that enables complex compiler transformations to be written, guarantees higher QoR after hardware mapping, and has built-in subcircuit deduplication support.
    Using this IR, we implement several compiler transformations targeting FPGA based RTL simulaton, as well as evaluate various subcircuit deduplication techniques and their tradeoffs.
  ],
  authors: (
    (
      name: "Joonho Whangbo",
      organization: [UC Berkeley],
      email: "joonho.whangbo@berkeley.edu"
    ),
    (
      name: "Maor Fuks",
      organization: [UC Berkeley],
      email: "maor_fuks@berkeley.du"
    ),
  ),
  index-terms: ("HDL", "RTL", "Compiler", "EDA/CAD"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction
A hardware intermediate representation (IR) has various benefits.
First of all, it enhances the reusability of designs specified using the frontend hardware description language (HDL) @firrtl.
Reusability is crucial as the design goes through various flows during design cycle.
At first, it is compiled into software RTL simulators to get the basic performance and functionality correct.
Once the design matures, it goes through FPGA flows for simulation and prototyping to iron out extra bugs as well as software/firmware development.
Finally, the design is pushed through the ASIC flow for tapeout.
By having a compiler stack and a IR, you can systematically manipulate the circuit so that it is suitable for different flows.
For instance, while ASIC memory blocks are notorious for getting replicated when mapped to FPGA BRAM/URAMs, a compiler can transform the design to prevent this replication.

Next, it makes instrumentation and debugging easier.
You can programatically insert scan-chains during your FPGA emulation flow, add coverage and assertions programatically, and perform preliminary correctness checks such as combinational loop detection.

Finally, it enables you to perform advanced simulation tricks.
One example is the FAME transformation @goldengate which lets you obtain cycle-exact simulation results, and perform resource optimizations for FPGA based RTL simulation.
Another example is deduplication or vectorization for software RTL simulation by identifying common subgraphs within a circuit.

In this report, we identify the shortcomings of the FIRRTL IR @firrtl and propose and implement new hardware IR abstractions. 
Using this IR, we implement several compiler transformations targeting FPGA based RTL simulaton, as well as evaluate various subcircuit deduplication techniques and their tradeoffs.

= FIRRTL Limitations
In this section we introduce the limitations of the FIRRTL IR and motivate why we need a new hardware IR abstraction.

== Statement based in-memory representation

#figure(
  image("./assets/firrtl.png", width: 100%),
  caption: [Example of an FIRRTL AST.],
) <fig:firrtl>

@fig:firrtl represents a FIRRTL AST which has a statement based in-memory representation.
This makes it easier for Chisel @chisel to generate the IR, and developers to read the textual form.
However, this simplicity comes at a cost because circuit traversals becomes a nightmare.
In order to traverse the graph, you must find the statements that has the current expression on the right-hand-side and search for connection/combinational statements.
To make matters worse, the AST is defined in a recursive manner: hence the compiler designer must write recursive functions to perform graph traversals.
This bad abstraction make analysis passes difficult to write, let alone the transformation passes.

== Immutable datastructure
FIRRTL is implemented as a immutable datastructure.
Hence, to perform circuit transformations, you must copy the entire IR into a new variable.
Coupled with having to perform recursion for compiler passes, these extra memcopies explodes the memory usage, frequently crashing the compiler for large designs.

== Eager expansion of behavioral constructs to structural constructs
In FIRRTL, behavioral constructs such as When/elsewhen/otherwise (think of if/else blocks in Verilog) are expanded eagerly to mux trees to simplify compiler passes.
This approach, while simplifying the compiler implementation, results in significant drawbacks.
Specifically, the generated Verilog code from these expanded constructs results in slower RTL simulation. Simulators are typically optimized to efficiently handle behavioral constructs like if/else statements. When these are expanded to structural mux trees (ternary statements in Verilog), simulators lose the ability to perform cycle-skipping optimizations, resulting in redundant computations.

== No support for circuit deduplication
FIRRTL lacks native support for circuit deduplication—the ability to efficiently identify common subcircuits. This limitation affects several critical aspects of modern hardware design flows.

Circuit deduplication is vital for incremental compilation, where only modified portions of a design need to be recompiled. Without this capability, even small changes require full recompilation of the entire design, significantly increasing turnaround time during the iterative development process. As designs grow in complexity, this limitation becomes increasingly problematic for developer productivity.

Additionally, circuit deduplication can be used for fine-grained FAME-5 @goldengate transformations. FAME-5 requires tracking of common subcircuits to optimize resource usage for FPGA-accelerated simulation. Without built-in deduplication support, implementing these transformations becomes cumbersome and error-prone, often requiring custom workarounds that are difficult to maintain and extend.

A proper deduplication mechanism would also enable more sophisticated version control integration, design evolution analysis, and targeted debugging of circuit changes—all capabilities that are standard in software development but remain challenging in hardware design due to these IR limitations.

== Miscellaneous
Another problem of FIRRTL is that it hardwires dont-cares to zeros, which limits the boolean optimization space, reducing QoR.
Also, FIRRTL doesn't support proper X-semantics which can lead to random bugs during simulation and post-silicon.


= New IR Abstractions

In this section, we elaborate on the new IR abstractions that we have implemented.
The compiler infrastructure is implemented using 11K lines of Rust.

== Graph based IR
Our new IR takes a fundamentally different approach by using a graph-based representation instead of a statement-based AST. This design choice offers several key advantages:

First, the graph structure makes it significantly easier to traverse and perform analysis passes. Unlike FIRRTL's recursive AST traversal, graph traversal can be done iteratively, reducing implementation complexity and stack overhead.

Second, better abstractions enable more complex transformations to be written with less effort. The graph structure naturally represents the actual connectivity of the circuit, making it easier to reason about and manipulate circuit topology.

This approach is similar to the successful redesign of Synopsys' Fusion Compiler. By redefining their software architecture, they enabled complex optimization flows that would have been prohibitively difficult to implement in their previous architecture. Our experience mirrors theirs – the graph-based approach has made previously complex transformations straightforward to implement.

== Defer eager lowering and hold on to semantics as long as possible

#figure(
  image("./assets/phinode.png", width: 50%),
  caption: [Phi nodes represents an abstract mux tree without converting the behavioral constructs into structural form.],
) <fig:phinode>

To address the limitations of FIRRTL's eager lowering, we introduce the phi-node abstraction as shown in @fig:phinode. Phi nodes provide a way to represent arbitrary combinational mux trees without converting behavioral constructs into their structural form. This abstraction allows us to:

1. Preserve high-level semantic information longer in the compilation pipeline
2. Enable more sophisticated optimizations that can reason about behavioral intent
3. Defer when-statement expansion until it's actually necessary
4. Maintain better correlation with the original source code

== Efficient Circuit Deduplication/Diffing
Circuit deduplication fundamentally boils down to solving the greatest common subgraph problem between two circuit graphs. While this problem is NP-complete in the general case, we recognize that hardware circuits have special properties that make practical solutions feasible. The ability to efficiently identify common subcircuits enables several key optimizations:

- Incremental compilation through smart reuse of previously compiled circuits
- Improved SW RTL simulation throughput by reducing instruction cache pressure
- Vectorization opportunities for parallel simulation of common subcircuits
- Automatic identification of FAME-5 candidates for FPGA-accelerated simulation

We explored two complementary approaches to this problem:

=== Approach 1 – Tree based circuit deduplication/diffing
Our first approach leverages the fact that we preserve assignment priorities within phi nodes, allowing us to reconstruct a tree representation from our graph IR. This enables us to build upon established work in AST diffing, specifically drawing from the GumTree algorithm @gumtree for fine-grained source code differencing.

The algorithm operates in two phases:
1. Top-down Phase: Identifies matching leaf nodes with height larger than a minimum threshold (h_min), establishing anchor points for further matching
2. Bottom-up Phase: Identifies similar nodes by considering both node attributes and the number of matching children, gradually building up larger matching subtrees

This approach is particularly effective for circuits that maintain a relatively tree-like structure, common in control logic and behavioral descriptions.

=== Approach 2 – Direct common subgraph identification
Our second approach is a greedy algorithm, executing the locally optimal move at each iteration @maximum_common_subgraph_for_digital_circuits. Given two IR graph abstractions g#sub[A]= (V#sub[A], E#sub[A], L#sub[A]) and g#sub[B]= (V#sub[B], E#sub[B], L#sub[B]), find a mapping M#sub[AB]: V#sub[A]#sym.arrow V#sub[B] that matches as many equivalent nodes as possible, the maximum common subgraph. We use a modified best-first search to heuristically and simultaneously traverse the graphs and identify common nodes, seeding from the input and output ports. The algorithm finds all adjacent vertices of the current node which are present in both graphs, then restricts the search space of candidate nodes by querying a previously-constructed LSH storage (see below) to only search pairs of nodes which are semantically and structurally similar. Then, the algorithm queries the best candidate through the LSH, maps it, and enqueues its undiscovered nodes.

=== Locality Sensitive Hashing (LSH)
We use locality sensitive hashing as a means to quickly identify semantically similar nodes. For every node we deterministically construct a vector of hashes of nodes within its k-neighborhood and store the vector in the LSH. The LSH type samples n (inputted parameter) points from the multivariate standard normal distribution, and projects each original vector onto a sampled point. These random projections preserve relative distance in expectation. The LSH then assigns the nodes to buckets determined by their Euclidean distance, meaning that similar nodes get mapped to the same bucket. This allows for constant-time retrieval of similar nodes.

Our second approach tackles the subgraph isomorphism problem directly, but exploits key properties of hardware circuits to make the problem tractable:

1. Circuit Sparsity: Hardware designs typically have bounded fan-in/fan-out, making the graph naturally sparse
2. Limited Cycles: Cycles in the graph are constrained to sequential logic elements, allowing us to partition the problem

We implement a hybrid algorithm @maximum_common_subgraph_for_digital_circuits that:
1. Uses locality sensitive hashing to identify candidate matching regions
2. Applies local pattern matching to expand matches

= Case Study: Implementing FAME transformations

As a case study, we implement the FAME-1 @goldengate transformation using our IR.

#figure(
  image("./assets/fame-1.png", width: 100%),
  caption: [
    FAME-1 decouples the target design's clock from the host FPGA clock, enabling simulations faithful to the taped-out ASIC on a FPGA.
    The black bold lines represents the original circuit, while the colored lines and elements represents the additional logic added by the compiler to perform host-clock decoupling.
  ],
) <fig:fame1>

The FAME-1 transformation decouples the target design's clock from the host FPGA clock by introducing clock-gating and stalling logic.
In our IR, this is implemented by identifying all clock domains and inserting control logic.
In our compiler, this pass is implemented in 421 lines of Rust compared to 474 lines of Scala.
Considering that Scala is a much expressive language, we can infer that the compiler passes are simpler to write.

= Evaluation

== Tree based circuit deduplication

#figure(
  image("./assets/gumtree-results.png", width: 100%),
  caption: [
    Evaluation of the GumTree algorithm on our example circuits.
  ],
) <fig:gumtree_results>

#figure(
  image("./assets/gumtree-time.png", width: 100%),
  caption: [
    Execution time of the GumTree algorithm on our example circuits.
  ],
) <fig:gumtree_time>

Our tree-based approach to circuit deduplication, while not performing structural or semantic diffing in the traditional sense, offers a practical solution to the subgraph matching problem. While we may not always find the largest possible common subgraph, our evaluation shows that the matches we do find are sufficient for practical optimization purposes.

The key insight is that for many hardware designs, especially those with regular structures and control logic, the tree representation captures enough of the circuit's structure to enable meaningful optimizations.
@fig:gumtree_results shows the results.
The match percentage represents the ratio of nodes that the algorithm identified as duplicated vs the original circuit.
For many cases, we can see that the algorithm works well, while there are cases where it fails catastrophically, and the match percentage is not necessarily correlated to the size of the circuit.
We leave the investigation of the algorithmic deficiencies to future work.

@fig:gumtree_time shows the execution time of the algorithm. We can see that it is linear to the size of the graph and can scale to larger graphs.

== Direct circuit deduplication


#figure(
  image("./assets/max_common_subgraph-results.png", width: 100%),
  caption: [
    Evaluation of the LSH maximum common subgraph algorithm on our example circuits.
  ],
) <fig:max_common_subgraph_results>

#figure(
  image("./assets/max_common_subgraph-time.png", width: 100%),
  caption: [
    Execution time of the LSH common subgraph algorithm on our example circuits.
  ],
) <fig:max_common_subgraph_time>

#figure(
  image("./assets/max_common_subgraph-k-vs-time.png", width: 100%),
  caption: [
    Execution time of the LSH common subgraph algorithm plotted against k, the size of each node's hash neighborhood.
  ],
) <fig:max_common_subgraph_k_vs_time>

Our approach to the maximum common subgraph problem provides a greedy algorithm attempting to make the optimal move at every step, arriving arbitrarily close to the global optimum. Figure 6 shows the results of the algorithm run on our example circuits, with a constant value k = 2 and 30-dimensional projections. We can see that the common subgraph identification remains accurate regardless of the size of the graphs. Comparing these to the GumTree algorithm, we can see that the results remain precise with some variance. Figure 7 shows the execution time plotted against the size of the graphs, scaling linearly in computational complexity @maximum_common_subgraph_for_digital_circuits and allowing for scalability. Figure 8 shows the execution time of the algorithm plotted against values of k. This figure resembles the logistic growth function, showcasing how increasing the value of k initially increases the execution time, and then stabilizes.

= Conclusion and Future Work

In this work, we have presented a new hardware IR that addresses several key limitations of existing HDL IRs.
Our graph-based representation, combined with deferred lowering of behavioral constructs and efficient circuit deduplication capabilities, enables more sophisticated compiler transformations and better quality of results.

Our evaluation shows that these improvements translate to practical benefits in terms of both compiler implementation complexity and the quality of generated hardware.

Looking forward, we see several directions for future work.
The holy grail of spatial + temporal FAME-5 transformations, combining pipelining with resource sharing, is now within reach thanks to our IR's flexible abstractions.
We also plan to explore more sophisticated circuit diffing techniques that could enable better version control and debugging capabilities for hardware designs.


= Role

Joonho worked on building the overall compiler infrastructure, the FAME-1 transformation, and the implementation of the GumTree algorithm.
Maor implemented the direct common subgraph identification algorithm.

= How the project applied one or more of the course topics

During class, we learned how dont-cares affects the QoR of your circuit.
We identified this as a problem of FIRRTL.
The approaches described attempt to reduce the complexity of the maximum common subgraph problem, with applications to FPGA simulation, partitioning, technology mapping, and more.
Furthermore, we have encountered many NP-complete problems throughout the course and learned you should still try to solve for the general case.
And indeed we did.
