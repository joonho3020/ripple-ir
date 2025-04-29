use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use petgraph::graph::NodeIndex;
use crate::common::graphviz::DefaultGraphVizCore;
use crate::ir::fir::FirGraph;
use crate::passes::fir::common_subgraph::neighbors_of_distance_k;
use lsh_rs::prelude::*;

pub fn get_hash_vector_of_distance_k(graph: &FirGraph, start: NodeIndex, k: usize) -> Vec<f32> {
    let neighbors = neighbors_of_distance_k(graph, start, k);
    let mut hashes:Vec<f32> = Vec::with_capacity(k);
    for v in neighbors {
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
    hashes.sort_by(|a, b| a.partial_cmp(b).unwrap()); // Sort by hash to be deterministic
    for i in 0..k {
        if hashes[i].is_nan() {
            hashes[i] = 0.0;
        }
    }
    return hashes
}
// Return LshMem of hashes of k-neighbors for all nodes in the graph, and vector identifying inserted order of nodes
pub fn get_k_lsh<T>(graph: &FirGraph, k: usize, n_projections: usize, n_hash_tables: usize) -> (LshMem<T, f32, i8>, Vec<NodeIndex>) {
    let mut lsh = LshMem::<T, f32, i8>::new(n_projections, n_hash_tables, k).l2().unwrap();
    let node_indices = graph.graph.node_indices().collect();
    let vec_of_k_neighbors_per_node:Vec<Vec<f32>> = node_indices.map(|&node_id| get_hash_vector_of_distance_k(graph, node_id, k)).collect(); // Map each node to its k-hash
    lsh.store_vecs(&vec_of_k_neighbors_per_node);
    return (lsh, vec_of_k_neighbors_per_node);
}

pub fn query_lsh<T>(graph: &FirGraph, lsh: LshMem<T, f32, i8>, node_ids:&Vec<NodeIndex>, starting_node:NodeIndex, k:usize) -> Vec<f32> {
    let node_indices = graph.graph.node_indices().collect();
    let vec_of_k_neighbors_per_node:Vec<f32> = get_hash_vector_of_distance_k(graph, starting_node, k);

}






// pub fn group_by_type<V, E>(graph: &Graph<V, E, Undirected>) -> Vec<Vec<NodeIndex>> {
//     let mut vec:HashMap<Type, Vec<NodeIndex>> = HashMap::new();
//     for node in graph.node_indices() {
//         if vec.contains_key(graph.typ)
//     }
// }