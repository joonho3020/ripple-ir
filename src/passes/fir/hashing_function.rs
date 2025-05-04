use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use chirrtl_parser::ast::Identifier;
use petgraph::graph::NodeIndex;
use crate::common::graphviz::DefaultGraphVizCore;
use crate::ir::fir::FirGraph;
use lsh_rs::{LshMem, L2};
use crate::passes::runner::run_passes_from_filepath;


pub fn neighbors_of_distance_k(graph: &FirGraph, start: NodeIndex, k: usize) -> Vec<NodeIndex> {
    let mut visited = HashSet::new(); // Dont backtrack
    let mut result = Vec::new();
    let mut current = Vec::new();
    visited.insert(start);
    result.push(start);
    current.push(start);
    for i in 0..k {
        let mut next = Vec::new();
        for neighbor in &current {
            for v in graph.graph.neighbors(*neighbor) {
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
            let src_vec = get_hash_vector_of_distance_k(src_fg, node_id, k); // Hash both
            let dst_vec = get_hash_vector_of_distance_k(dst_fg, node_id, k);
            let bucket: Vec<&Vec<f32>> = lsh.query_bucket(&dst_vec).unwrap(); // Bucket which dst_vec is in

            if !bucket.contains(&&src_vec) { // Didnt land in same bucket
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
            let weight = fir_graph.graph.node_weight(id).unwrap();
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



#[cfg(test)]
mod tests {
    use graphviz_rust::attributes::color_name::darkblue;
    use super::*;
    const k: usize = 5;
    const n_projections: usize = 15;
    const n_hash_tables: usize = 10;
    const bucket_width: f32 = 0.2;

    #[test]
    pub fn test_GCD() {
        let src_file:&str = "GCD";
        let dst_file:&str = "GCDDelta";
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
}



