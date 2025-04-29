use std::collections::HashSet;
use petgraph::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::graph::EdgeIndex;
use petgraph::unionfind::UnionFind;
use petgraph::Undirected;
use priority_queue::PriorityQueue;
use petgraph::visit::{EdgeRef, IntoNeighbors};
use crate::ir::fir::FirGraph;

// Return minimum spanning tree using Kruskals algorithm, (v1, v2) is an undirected edge between v1 and v2
pub fn graph_kruskals(graph: &Graph<u32, u32, Undirected>) -> Vec<(NodeIndex, NodeIndex)> {
    let mut uf = UnionFind::new(graph.node_count());
    let mut edges_vec: Vec<EdgeIndex> = graph.edge_indices().collect();
    let mut minimum_spanning_tree = Vec::new();
    for i in 0..edges_vec.len() {
        let mut min = i;
        for j in (i + 1)..edges_vec.len() {
            if graph.edge_weight(edges_vec[j]).unwrap() < graph.edge_weight(edges_vec[min]).unwrap() {
                min = j;
            }
        }
        let temp = edges_vec[i];
        edges_vec[i] = edges_vec[min];
        edges_vec[min] = temp;
    }
    for edge_id in edges_vec {
        let (v1, v2) = graph.edge_endpoints(edge_id).unwrap();
        if uf.find(v1.index()) != uf.find(v2.index()) {
            uf.union(v1.index(), v2.index());
            minimum_spanning_tree.push((v1, v2));
        }
    }
    return minimum_spanning_tree
}
// Return maximum spanning tree using prims algorithm, of type Vec<(vertex_1, vertex_2)>, where each (v1, v2) is an edge
pub fn graph_prims(graph: &Graph<u32, u32, Undirected>, start_vertex: NodeIndex) -> Vec<(NodeIndex, NodeIndex)> {
    let mut visited: HashSet<NodeIndex> = HashSet::new();
    let mut pq = PriorityQueue::new(); // ((v1, v2), w) = (edge, weight)
    let mut maximum_spanning_tree = Vec::new();
    visited.insert(start_vertex);
    for edge in graph.edges(start_vertex) {
        pq.push((start_vertex, edge.target()), *edge.weight());
    }
    while !pq.is_empty() {
        let ((v1, v2), weight) = pq.pop().unwrap();
        if !visited.contains(&v2) {
            visited.insert(v2);
            if v1 < v2 {
                maximum_spanning_tree.push((v1, v2));
            }
            else if v1 > v2 {
                maximum_spanning_tree.push((v2, v1));
            }
            for edge in graph.edges(v2) {
                let neighbor = edge.target();
                if !visited.contains(&neighbor) {
                    pq.push((v2, neighbor), *edge.weight());
                }
            }
        }
    }
    return maximum_spanning_tree
}

// BFS for nodes of exactly distance k away from start node
pub fn neighbors_of_distance_k(graph: &FirGraph, start: NodeIndex, k: usize) -> Vec<NodeIndex> {
    let mut visited = HashSet::new(); // Dont backtrack
    let mut vec = Vec::new();
    vec.push(start);
    for i in 0..k {
        let mut curr_vec = Vec::new();
        for neighbor in &vec {
            for v in graph.graph.neighbors(*neighbor) {
                if !visited.contains(&v) {
                    visited.insert(v);
                    curr_vec.push(v);
                }
            }
        }
        vec = curr_vec;
    }
    return vec
}








#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kruskals1() {
        let mut g = Graph::<u32, u32, Undirected>::new_undirected();
        let n0 = g.add_node(0);
        let n1 = g.add_node(1);
        let n2 = g.add_node(2);
        let n3 = g.add_node(3);
        let n4 = g.add_node(4);
        let n5 = g.add_node(5);
        let n6 = g.add_node(6);
        let n7 = g.add_node(7);
        let n8 = g.add_node(8);

        g.add_edge(n0, n7, 8);
        g.add_edge(n0, n1, 4);
        g.add_edge(n1, n2, 8);
        g.add_edge(n2, n3, 7);
        g.add_edge(n3, n4, 9);
        g.add_edge(n4, n5, 10);
        g.add_edge(n5, n6, 2);
        g.add_edge(n6, n7, 1);
        g.add_edge(n1, n7, 11);
        g.add_edge(n7, n8, 7);
        g.add_edge(n6, n8, 6);
        g.add_edge(n2, n8, 2);
        g.add_edge(n2, n5, 4);
        g.add_edge(n3, n5, 14);

        let result:HashSet<_> = graph_kruskals(&g).into_iter().collect();
        let correct:HashSet<_> = vec![(n0, n1), (n0, n7), (n6, n7), (n5, n6), (n2, n5), (n2, n8), (n2, n3), (n3, n4)].into_iter().collect();
        assert_eq!(result, correct);
    }

    #[test]
    fn test_kruskals2() {
        let mut g = Graph::<u32, u32, Undirected>::new_undirected();
        let n0 = g.add_node(0);
        let n1 = g.add_node(1);
        let n2 = g.add_node(2);
        let n3 = g.add_node(3);
        let n4 = g.add_node(4);
        let n5 = g.add_node(5);
        let n6 = g.add_node(6);
        let n7 = g.add_node(7);
        let n8 = g.add_node(8);

        g.add_edge(n0, n7, 8);
        g.add_edge(n0, n1, 4);
        g.add_edge(n1, n2, 8);
        g.add_edge(n2, n3, 7);
        g.add_edge(n3, n4, 9);
        g.add_edge(n4, n5, 10);
        g.add_edge(n5, n6, 2);
        g.add_edge(n6, n7, 1);
        g.add_edge(n1, n7, 11);
        g.add_edge(n7, n8, 7);
        g.add_edge(n6, n8, 6);
        g.add_edge(n2, n8, 2);
        g.add_edge(n2, n5, 4);
        g.add_edge(n3, n5, 1);

        let result:HashSet<_> = graph_kruskals(&g).into_iter().collect();
        let correct:HashSet<_> = vec![(n0, n1), (n0, n7), (n2, n5), (n2, n8), (n3, n5), (n3, n4), (n5, n6), (n6, n7)].into_iter().collect();
        assert_eq!(result, correct);
    }

    #[test]
    fn test_tree1() {
        let mut g = Graph::<u32, u32, Undirected>::new_undirected();
        let n0 = g.add_node(0);
        let n1 = g.add_node(0);
        let n2 = g.add_node(0);
        let n3 = g.add_node(0);
        let n4 = g.add_node(0);
        let n5 = g.add_node(0);
        let n6 = g.add_node(0);
        let n7 = g.add_node(0);
        let n8 = g.add_node(0);
        let n9 = g.add_node(0);
        let n10 = g.add_node(0);
        let n11 = g.add_node(0);
        let n12 = g.add_node(0);
        let n13 = g.add_node(0);
        let n14 = g.add_node(0);
        let n15 = g.add_node(0);

        g.add_edge(n0, n1, 1);
        g.add_edge(n1, n2, 1);
        g.add_edge(n2, n10, 1);
        g.add_edge(n9, n15, 1);
        g.add_edge(n7, n8, 1);
        g.add_edge(n7, n13, 1);
        g.add_edge(n13, n14, 1);
        g.add_edge(n12, n13, 1);
        g.add_edge(n11, n12, 1);
        g.add_edge(n3, n12, 1);
        g.add_edge(n4, n12, 1);
        g.add_edge(n4, n5, 1);
        let result:HashSet<_> = graph_prims(&g, n8).into_iter().collect();
        let correct:HashSet<_> = vec![(n7, n8), (n7, n13), (n13, n14), (n12, n13), (n11, n12), (n3, n12), (n4, n12), (n4, n5)].into_iter().collect();
        assert_eq!(result, correct);
    }

    #[test]
    fn test_tree2() {
        let mut g = Graph::<u32, u32, Undirected>::new_undirected();
        let n0 = g.add_node(0);
        let n1 = g.add_node(0);
        let n2 = g.add_node(0);
        let n3 = g.add_node(0);
        let n4 = g.add_node(0);
        let n5 = g.add_node(0);
        let n6 = g.add_node(0);
        let n7 = g.add_node(0);
        let n8 = g.add_node(0);
        let n9 = g.add_node(0);
        let n10 = g.add_node(0);
        let n11 = g.add_node(0);
        let n12 = g.add_node(0);
        let n13 = g.add_node(0);
        let n14 = g.add_node(0);
        let n15 = g.add_node(0);

        g.add_edge(n0, n1, 1);
        g.add_edge(n1, n2, 1);
        g.add_edge(n2, n10, 1);
        g.add_edge(n9, n15, 1);
        g.add_edge(n7, n8, 1);
        g.add_edge(n7, n13, 1);
        g.add_edge(n13, n14, 1);
        g.add_edge(n12, n13, 1);
        g.add_edge(n11, n12, 1);
        g.add_edge(n3, n12, 1);
        g.add_edge(n4, n12, 1);
        g.add_edge(n4, n5, 1);
        g.add_edge(n10, n11, 1);
        let result:HashSet<_> = graph_prims(&g, n12).into_iter().collect();
        let correct:HashSet<_> = vec![(n7, n8), (n7, n13), (n13, n14), (n12, n13), (n11, n12), (n3, n12), (n4, n12), (n4, n5), (n10, n11), (n0, n1), (n1, n2), (n2, n10)].into_iter().collect();
        assert_eq!(result, correct);
    }
}