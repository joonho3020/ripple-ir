use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use chirrtl_parser::ast::Identifier;
use petgraph::graph::NodeIndex;
use crate::common::graphviz::DefaultGraphVizCore;
use crate::ir::fir::{FirGraph, FirNodeType};
use lsh_rs::{LshMem, L2};
use petgraph::data::DataMap;
use petgraph::visit::IntoNeighbors;
use crate::passes::runner::run_passes_from_filepath;
use std::time::Instant;


pub fn neighbors_of_distance_k(graph: &FirGraph, start: NodeIndex, k: usize) -> Vec<NodeIndex> {
    let mut visited = BTreeSet::new(); // Dont backtrack
    let mut result = Vec::new();
    let mut current = Vec::new();
    visited.insert(start);
    result.push(start);
    current.push(start);
    for i in 0..k {
        let mut next = Vec::new();
        for neighbor in &current {
            for v in graph.graph.neighbors_undirected(*neighbor) {
                if !visited.contains(&v) {
                    visited.insert(v);
                    result.push(v);
                    next.push(v);
                }
            }
        }

        if next.is_empty() {
            break;
        }

        current = next;
    }
    return result;
}

// Return vector of hashes of all k-neighbors of start
pub fn get_hash_vector_of_distance_k(graph: &FirGraph, start: NodeIndex, k: usize) -> Vec<f32> {
    let neighbors = neighbors_of_distance_k(graph, start, k);
    let mut hashes:Vec<f32> = Vec::with_capacity(k);
    for v in neighbors {

        if graph.node_weight(v).is_none() {
            continue;
        }

        let neighbor_type = &graph.node_weight(v).unwrap().nt; // FirNodeType
        let neighbor_name = &graph.node_weight(v).unwrap().name;  // Identifier
        let mut hasher = DefaultHasher::new(); // Hashing function
        neighbor_type.hash(&mut hasher);

        if !neighbor_name.is_none() {
            neighbor_name.hash(&mut hasher);
        }
        let final_hash = hasher.finish();
        hashes.push(final_hash as f32);
    }
    hashes.sort_by(|a, b| a.partial_cmp(b).unwrap()); // Sort hashes to be deterministic
    hashes.resize(k, 0.0); // Fill rest with 0.0
    for x in &mut hashes {
        if x.is_nan() {
            *x = 0.0;
        }
    }
    return hashes
}


// Return LshMem of hashes of k-neighbors for all nodes in the graph, and vector identifying inserted order of nodes
pub fn get_k_lsh(graph: &FirGraph, k: usize, n_projections: usize, n_hash_tables: usize, bucket_width: f32) -> (LshMem<L2>, Vec<NodeIndex>) {
    let node_indices:Vec<NodeIndex> = graph.graph.node_indices().collect(); // Insertion order, return to be deterministic
    let mut vec_of_k_neighbors_per_node:Vec<Vec<f32>> = Vec::with_capacity(node_indices.len());

    for &id in &node_indices { // Map each node to its k-hash
        let vec:Vec<f32> = get_hash_vector_of_distance_k(&graph, id, k);
        vec_of_k_neighbors_per_node.push(vec);
    }

    let mut lsh = LshMem::<L2>::new(n_projections, n_hash_tables, k).l2(bucket_width).unwrap(); // Define LSH
    lsh.store_vecs(&vec_of_k_neighbors_per_node);
    return (lsh, node_indices);
}

// Input two file names, return HashMap with entries (module name, vector of differing nodes) of differing nodes
pub fn compare_graphs(src_file: &str, dst_file: &str, k: usize, n_proj: usize, n_tables: usize, bucket_width: f32) -> HashMap<Identifier, Vec<NodeIndex>> {
    let src_fir = run_passes_from_filepath(&format!("./test-inputs/{}.fir", src_file)).expect("failed src file");
    let dst_fir = run_passes_from_filepath(&format!("./test-inputs/{}.fir", dst_file)).expect("failed dst file");
    let mut src_list:Vec<(&Identifier, &FirGraph)> = src_fir.graphs.iter().collect(); // Collect all modules of FirIR
    let mut dst_list:Vec<(&Identifier, &FirGraph)> = dst_fir.graphs.iter().collect();
    let mut different_nodes:HashMap<Identifier, Vec<NodeIndex>> = HashMap::new();

    // Iterate through all pairs of vertices in opposing graphs, 'zip' tuples together, automatically stops at min(src.len(), dst.len())
    for ((src_name, src_fg), (dst_name, dst_fg)) in src_list.iter().zip(dst_list.iter()) {
        // Construct a LSH for the src FirGraph
        let (lsh, node_order) = get_k_lsh(src_fg, k, n_proj, n_tables, bucket_width);
        let mut missing: Vec<NodeIndex> = Vec::new();

        for &node_id in node_order.iter() { // Same as inserted order
            if dst_fg.graph.node_weight(node_id).is_none() {
                missing.push(node_id);
                continue;
            }
            if src_fg.graph.node_weight(node_id).unwrap().nt != dst_fg.graph.node_weight(node_id).unwrap().nt {
                missing.push(node_id);
                continue;
            }

            let src_vec = get_hash_vector_of_distance_k(src_fg, node_id, k); // Hash both
            let dst_vec = get_hash_vector_of_distance_k(dst_fg, node_id, k);
            let bucket: Vec<&Vec<f32>> = lsh.query_bucket(&dst_vec).unwrap(); // Bucket which dst_vec is in
            let pointer = &src_vec;

            if !bucket.contains(&pointer) { // Didnt land in same bucket
                missing.push(node_id);
            }
        }
        if !missing.is_empty() {
            different_nodes.insert((*src_name).clone(), missing);
        }
    }
    return different_nodes;
}

// Print all nodes that did not land in the same bucket, input HashMap of differing nodes from compare_graphs()
pub fn print_differing_nodes(map: &HashMap<Identifier, Vec<NodeIndex>>, src_file: &str) {
    if map.is_empty() { // No nodes differ
        return;
    }
    let src_fir = run_passes_from_filepath(&format!("./test-inputs/{}.fir", src_file)).expect("src file doesn't exist"); // To retrieve Identifier/name of nodes

    for (graph_name, nodes) in map { // For every graph/module
        let fir_graph = src_fir.graphs.get(graph_name).unwrap();
        eprintln!("Graph/module: '{}'", graph_name);
        eprintln!("   Differing nodes: ");

        for &id in nodes {
            let weight = fir_graph.graph.node_weight(id).unwrap(); // Get name of node
            let node_name:String = if let Some(name) = &weight.name { // If node has a name (type Identifier)
                name.to_string()
            } else {
                format!("      Unnamed node with ID: {:?}", id)
            };

            if weight.name.is_none() {
                eprintln!("{}", node_name);
            }
            else {
                eprintln!("      {} (ID = {:?})", node_name, id);
            }
        }
    }
}


// Return ports in deterministic order, (inputs, outputs)
pub fn find_ports(graph:&FirGraph) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
    fn comparator(graph:&FirGraph, node1:NodeIndex, node2:NodeIndex) -> Ordering { // Compare by name first, then NodeIndex;
        let w1 = graph.node_weight(node1).unwrap();
        let w2 = graph.node_weight(node2).unwrap();
        match (w1.name.as_ref(), w2.name.as_ref()) {
            (Some(name1), Some(name2)) => name1.cmp(name2).then_with(|| node1.index().cmp(&node2.index())),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => node1.index().cmp(&node2.index()),
        }
    }
    let mut inputs:Vec<NodeIndex> = Vec::new();
    let mut outputs:Vec<NodeIndex> = Vec::new();
    for i in graph.node_indices() {
        let node = graph.node_weight(i).unwrap();
        match node.nt {
            FirNodeType::Input => {inputs.push(i);},
            FirNodeType::Output => {outputs.push(i)}
            _ => {continue}
        }
    }
    inputs.sort_by(|&a, &b| comparator(graph, a, b));
    outputs.sort_by(|&a, &b| comparator(graph, a, b));
    (inputs, outputs)
}

pub fn have_all_common_neighbors(graph:&FirGraph, node:NodeIndex, map:&Vec<NodeIndex>) -> bool {
    let all_neighbors:HashSet<NodeIndex> = graph.graph.neighbors_undirected(node).collect();
    if map.iter().all(|n| all_neighbors.contains(n)) {
        return true;
    }
    return false;
}

pub fn vector_distance(vec1:&Vec<f32>, vec2:&Vec<f32>) -> f32 {
    let mut distance = 0.0;
    for (x1, x2) in vec1.iter().zip(vec2.iter()) {
        distance = distance + (x1 - x2).abs();
    }
    return distance
}

// Find closest candidate in bucket
pub fn find_best_candidate(nodes:Vec<NodeIndex>, graph: &FirGraph, query_vec:&Vec<f32>, k:usize) -> NodeIndex {
    let mut best:NodeIndex = nodes[0];
    let mut closest_distance = vector_distance(query_vec, &get_hash_vector_of_distance_k(graph, best, k));
    for i in 1..nodes.len() {
        let distance = vector_distance(query_vec, &get_hash_vector_of_distance_k(graph, nodes[i], k));
        if distance < closest_distance {
            closest_distance = distance;
            best = nodes[i];
        }
    }
    return best;
}


pub fn max_common_subgraph(graphA: &FirGraph, graphB: &FirGraph, k:usize, n_proj:usize, n_tables:usize, bucket_width:f32) -> HashMap<NodeIndex, NodeIndex> {
    let (in_A, out_A) = find_ports(graphA);
    let (in_B, out_B) = find_ports(graphB);
    let mut in_name:HashMap<Identifier, NodeIndex> = HashMap::new();
    for &id in &in_B { // Only matching input ports
        if let Some(ident) = graphB.node_weight(id).unwrap().name.clone() {
            in_name.insert(ident, id);
        }
    }
    let mut out_name:HashMap<Identifier, NodeIndex> = HashMap::new();
    for &id in &out_B { // Only matching output ports
        if let Some(ident) = graphB.node_weight(id).unwrap().name.clone() {
            out_name.insert(ident, id);
        }
    }
    let mut max_common_subgraph:HashMap<NodeIndex, NodeIndex> = HashMap::new(); // To return
    let mut F = BTreeSet::new(); // Final
    for &id in &in_A{ // Match inputs
        if let Some(identA) = graphA.node_weight(id).unwrap().name.as_ref() {
            if let Some(&identB) = in_name.get(identA) {
                max_common_subgraph.insert(id, identB);
                F.insert(id);
            }
        }
    }
    for &id in &out_A{ // Match outputs
        if let Some(identA) = graphA.node_weight(id).unwrap().name.as_ref() {
            if let Some(&identB) = out_name.get(identA) {
                max_common_subgraph.insert(id, identB);
                F.insert(id);
            }
        }
    }
    for idx_a in graphA.node_indices() {
        let Some(name_a) = graphA.node_weight(idx_a).unwrap().name.as_ref() else { //
            continue
        };
        if max_common_subgraph.contains_key(&idx_a) { // Already mapped
            continue
        };
        if let Some(idx_b) = graphB.node_indices().find(|&n| graphB.node_weight(n).unwrap().name.as_ref() == Some(name_a)) { // If exists in both graphs
            if graphA.node_weight(idx_a).unwrap().nt == graphB.node_weight(idx_b).unwrap().nt { // Same name
                max_common_subgraph.insert(idx_a, idx_b);
                F.insert(idx_a);
            }
        }
    }
    let mut D: VecDeque<NodeIndex> = VecDeque::new(); // Initialize discovered
    for &v in &F {
        for neighbor in graphA.graph.neighbors_undirected(v) {
            if !F.contains(&neighbor) && !D.contains(&neighbor) {
                D.push_back(neighbor);
            }
        }
    }
    let mut blocked:HashSet<NodeIndex> = HashSet::new();
    let (lshB, orderB) = get_k_lsh(graphB, k, n_proj, n_tables, bucket_width); // LSH
    while let Some(dA) = D.pop_front() { // While D not null set
        if blocked.contains(&dA) {
            continue;
        }
        let mut NB:Vec<NodeIndex> = Vec::new(); // Neighbor candidates
        for v in graphA.graph.neighbors_undirected(dA) {
            if max_common_subgraph.get(&v).is_some() {
                NB.push(v);
            }
        }
        let mut filtered_NB = Vec::new();
        for n in &NB {
            if let Some(&v) = max_common_subgraph.get(n) {
                filtered_NB.push(v);
            }
        }
        if filtered_NB.is_empty() {
            blocked.insert(dA);
            continue;
        }
        let hash_vector = get_hash_vector_of_distance_k(graphA, dA, k);
        let mut CB = Vec::new();
        for &nodeB in orderB.iter() {
            let hash_vec = get_hash_vector_of_distance_k(graphB, nodeB, k);
            if graphB.node_weight(nodeB).unwrap().nt != graphA.node_weight(dA).unwrap().nt { // If different FirNodeType, continue
                continue;
            }
            if max_common_subgraph.values().any(|&mapped| mapped == nodeB) { // If already mapped, continue
                continue;
            }
            if have_all_common_neighbors(&graphB, nodeB, &filtered_NB) { // Have all identical neighbors
                CB.push(nodeB);
            }
        }
        if !CB.is_empty() {
            let best_candidate = find_best_candidate(CB, &graphB, &hash_vector, k); // Best candidate heuristic
            max_common_subgraph.insert(dA, best_candidate);
            for neighbor in graphA.graph.neighbors_undirected(dA) {
                if !F.contains(&neighbor) && !D.contains(&neighbor) && !blocked.contains(&neighbor) {
                    D.push_back(neighbor);
                }
            }
            F.insert(dA);
        }
        else {
            blocked.insert(dA); // Not suitable
        }

    }
    return max_common_subgraph
}



pub fn max_approx(src_file: &str, dst_file: &str, k: usize, n_proj: usize, n_tables: usize, bucket_width: f32) -> HashMap<Identifier, Vec<NodeIndex>> {
    let diffs = compare_graphs(src_file, dst_file, k, n_proj, n_tables, bucket_width); // Different nodes
    let src_fir = run_passes_from_filepath(&format!("./test-inputs/{}.fir", src_file)).expect("failed to load src .fir");
    let mut common_map = HashMap::new();
    for (mod_name, src_fg) in src_fir.graphs.iter() {
        let (in_ports, out_ports) = find_ports(src_fg);
        let mut common_nodes: Vec<NodeIndex> = if let Some(missing) = diffs.get(mod_name) {
            src_fg.graph.node_indices().filter(|id| !missing.contains(id)).collect() // All nodes not differing
        } else {
            src_fg.graph.node_indices().collect()
        };
        for &port in in_ports.iter().chain(out_ports.iter()) { // Inputs and outputs
            if !common_nodes.contains(&port) {
                common_nodes.push(port);
            }
        }
        if !common_nodes.is_empty() {
            common_map.insert(mod_name.clone(), common_nodes);
        }
    }
    return common_map
}















#[cfg(test)]
mod tests {
    use petgraph::visit::NodeCount;
    use super::*;
    const k: usize = 2;
    const n_projections: usize = 20;
    const n_hash_tables: usize = 16;
    const bucket_width: f32 = 1.0;

    fn node_desc(g: &FirGraph, idx: NodeIndex) -> String {
        let w = g.graph.node_weight(idx).unwrap();
        let label = format!("{:?}", w.nt);
        let name  = w.name.as_ref().map(|id| id.to_string()).unwrap_or_else(|| format!("unnamed_{}", idx.index()));
        format!("{}[{}]", name, label)
    }

    #[test]
    pub fn test_GCD() {
        let src_file:&str = "GCD";
        let dst_file:&str = "GCDDelta";
        let map = compare_graphs(src_file, dst_file, k, n_projections, n_hash_tables, bucket_width);
        print_differing_nodes(&map, src_file);
        assert!(map.is_empty(), "Circuits are different");
    }

    #[test]
    pub fn test_BitSel() {
        let src_file:&str = "BitSel1";
        let dst_file:&str = "BitSel2";
        let map = compare_graphs(src_file, dst_file, k, n_projections, n_hash_tables, bucket_width);
        print_differing_nodes(&map, src_file);
        assert!(map.is_empty(), "Circuits are different");
    }

    #[test]
    pub fn test_AESStep() {
        let src_file:&str = "AESStep1";
        let dst_file:&str = "AESStep2";
        let map = compare_graphs(src_file, dst_file, k, n_projections, n_hash_tables, bucket_width);
        print_differing_nodes(&map, src_file);
        assert!(map.is_empty(), "Circuits are different");
    }

    #[test]
    pub fn test_IssueSlot() {
        let src_file:&str = "IssueSlot_1";
        let dst_file:&str = "IssueSlot_24";
        let map = compare_graphs(src_file, dst_file, k, n_projections, n_hash_tables, bucket_width);
        print_differing_nodes(&map, src_file);
        assert!(map.is_empty(), "Circuits are different");
    }

    #[test]
    pub fn test_CordicStep() {
        let src_file:&str = "CordicStep1";
        let dst_file:&str = "CordicStep3";
        let map = compare_graphs(src_file, dst_file, k, n_projections, n_hash_tables, bucket_width);
        print_differing_nodes(&map, src_file);
        assert!(map.is_empty(), "Circuits are different");
    }

    #[test]
    pub fn test_FFTStep() {
        let src_file:&str = "FFTStep1";
        let dst_file:&str = "FFTStep4";
        let map = compare_graphs(src_file, dst_file, k, n_projections, n_hash_tables, bucket_width);
        print_differing_nodes(&map, src_file);
        assert!(map.is_empty(), "Circuits are different");
    }



    #[test]
    pub fn test_approx() {
        let start = Instant::now();
        let mcs = max_approx("AESStep2", "AESStep3", k, n_projections, n_hash_tables, bucket_width);
        let src_fir = run_passes_from_filepath(&format!("./test-inputs/{}.fir", "BitSel1")).expect("failed to load src .fir");
        for (module, nodes) in mcs {
            let (i, graph) = src_fir.graphs.iter().next().unwrap();
            eprintln!("Module {} has {} common nodes", module, nodes.len());
            eprintln!("Match percentage: {}", (nodes.len() as f32) / (graph.node_indices().count() as f32));
            for node in nodes {
                eprintln!("{:#?}", node);
            }
        }
        eprintln!("Elapsed time: {:#?}", start.elapsed().as_millis());
    }

    #[test]
    fn test_mcs_and_print1() {
        let start = Instant::now();
        let src_fir = run_passes_from_filepath("./test-inputs/GCD.fir").expect("failed to load GCD.fir");
        let dst_fir = run_passes_from_filepath("./test-inputs/GCDDelta.fir").expect("failed to load GCDDelta.fir");
        let src_graph = src_fir.graphs.values().next().unwrap();
        let dst_graph = dst_fir.graphs.values().next().unwrap();
        let mapping = max_common_subgraph(src_graph, dst_graph, k, n_projections, n_hash_tables, bucket_width);
        assert!(!mapping.is_empty(), "MCS mapping should not be empty");
        let covered = mapping.len();
        let total   = src_graph.graph.node_count();
        eprint!("≈‑MCS size: {} of {} source vertices\n", covered, total);
        eprintln!("Percent mapped: {:.2}", (covered as f32) / (total as f32) * 100.0);
        for (src_idx, dst_idx) in &mapping {
            eprintln!("{:<40}  ↔  {}", node_desc(src_graph, *src_idx), node_desc(dst_graph, *dst_idx));
        }
        eprintln!("Elapsed time: {:#?}", start.elapsed().as_millis());
    }

    #[test]
    fn test_mcs_and_print2() {
        let src_fir = run_passes_from_filepath("./test-inputs/GCD.fir").expect("failed to load GCD.fir");
        let dst_fir = run_passes_from_filepath("./test-inputs/GCDDelta.fir").expect("failed to load GCDDelta.fir");
        let src_graph = src_fir.graphs.values().next().unwrap();
        let dst_graph = dst_fir.graphs.values().next().unwrap();
        let mapping = max_common_subgraph(src_graph, dst_graph, k, n_projections, n_hash_tables, bucket_width);
        assert!(!mapping.is_empty(), "MCS mapping should not be empty");
        let covered = mapping.len();
        let total   = src_graph.graph.node_count();
        eprintln!("≈‑MCS size: {} of {} source vertices\n", covered, total);
        for (src_idx, dst_idx) in &mapping {
            eprintln!("{:<40}  ↔  {}", node_desc(src_graph, *src_idx), node_desc(dst_graph, *dst_idx)
            );
        }
        eprintln!();
    }
}



