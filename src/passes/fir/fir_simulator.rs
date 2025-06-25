use crate::ir::fir::{FirGraph, FirNodeType};
use num_traits::ToPrimitive;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, VecDeque};
use rusty_firrtl::Int;

#[derive(Debug, Clone, PartialEq)]
pub enum FirValue {
    X,
    Int(Int),
}

impl FirValue {
    pub fn to_int(&self) -> Option<Int> {
        match self {
            FirValue::Int(i) => Some(i.clone()),
            _ => None,
        }
    }
}

pub struct FirSimulator {
    pub graph: FirGraph,
    pub values: HashMap<NodeIndex, FirValue>,
}

impl FirSimulator {
    pub fn new(graph: FirGraph) -> Self {
        Self {
            graph,
            values: HashMap::new(),
        }
    }

    pub fn set_input(&mut self, name: &str, value: Int) {
        for node_idx in self.graph.graph.node_indices() {
            let node = &self.graph.graph[node_idx];
            if let Some(n) = &node.name {
                if n.to_string() == name {
                    self.values.insert(node_idx, FirValue::Int(value.clone()));
                }
            }
        }
    }

    pub fn run(&mut self) {
        let mut queue: VecDeque<NodeIndex> = VecDeque::new();
        let mut in_degree = HashMap::new();
        // Initialize in-degree
        for node_idx in self.graph.graph.node_indices() {
            let indeg = self.graph.graph.edges_directed(node_idx, petgraph::Direction::Incoming).count();
            in_degree.insert(node_idx, indeg);
            if indeg == 0 {
                queue.push_back(node_idx);
            }
        }
        // Topological order simulation
        while let Some(node_idx) = queue.pop_front() {
            let node = &self.graph.graph[node_idx];
            let value = match &node.nt {
                FirNodeType::UIntLiteral(_, val) => FirValue::Int(val.clone()),
                FirNodeType::Input => self.values.get(&node_idx).cloned().unwrap_or(FirValue::X),
                FirNodeType::Reg => {
                    // For this example, treat reg as 0-initialized if not specified
                    self.values.get(&node_idx).cloned().unwrap_or(FirValue::Int(Int::from(0u32)))
                }
                FirNodeType::PrimOp2Expr(op) => {
                    let mut operands = vec![];
                    for edge in self.graph.graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                        let src = edge.source();
                        if let Some(FirValue::Int(val)) = self.values.get(&src) {
                            operands.push(val.clone());
                        } else {
                            operands.push(Int::from(0u32));
                        }
                    }
                    if operands.len() == 2 {
                        let a = operands[0].0.to_i64().unwrap();
                        let b = operands[1].0.to_i64().unwrap();
                        match op {
                            rusty_firrtl::PrimOp2Expr::And => FirValue::Int(Int::from((a & b) as u32)),
                            rusty_firrtl::PrimOp2Expr::Or => FirValue::Int(Int::from((a | b) as u32)),
                            rusty_firrtl::PrimOp2Expr::Eq => FirValue::Int(Int::from((a == b) as u32)),
                            rusty_firrtl::PrimOp2Expr::Neq => FirValue::Int(Int::from((a != b) as u32)),
                            rusty_firrtl::PrimOp2Expr::Lt => FirValue::Int(Int::from((a < b) as u32)),
                            rusty_firrtl::PrimOp2Expr::Leq => FirValue::Int(Int::from((a <= b) as u32)),
                            rusty_firrtl::PrimOp2Expr::Gt => FirValue::Int(Int::from((a > b) as u32)),
                            rusty_firrtl::PrimOp2Expr::Geq => FirValue::Int(Int::from((a >= b) as u32)),
                            rusty_firrtl::PrimOp2Expr::Add => FirValue::Int(Int::from((a + b) as u32)),
                            rusty_firrtl::PrimOp2Expr::Sub => FirValue::Int(Int::from((a - b) as u32)),
                            _ => FirValue::X,
                        }
                    } else {
                        FirValue::X
                    }
                }
                FirNodeType::Output => {
                    // Output just forwards its input
                    let mut val = FirValue::X;
                    for edge in self.graph.graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                        let src = edge.source();
                        if let Some(v) = self.values.get(&src) {
                            val = v.clone();
                            break;
                        }
                    }
                    val
                }
                FirNodeType::SMem(_) | FirNodeType::CMem => {
                    // For demo, treat as X
                    FirValue::X
                }
                _ => FirValue::X,
            };
            self.values.insert(node_idx, value);
            // Decrement in-degree of neighbors
            for edge in self.graph.graph.edges_directed(node_idx, petgraph::Direction::Outgoing) {
                let tgt = edge.target();
                if let Some(e) = in_degree.get_mut(&tgt) {
                    *e -= 1;
                    if *e == 0 {
                        queue.push_back(tgt);
                    }
                }
            }
        }
    }

    pub fn get_output(&self, name: &str) -> Option<FirValue> {
        for node_idx in self.graph.graph.node_indices() {
            let node = &self.graph.graph[node_idx];
            if let Some(n) = &node.name {
                if n.to_string() == name {
                    return self.values.get(&node_idx).cloned();
                }
            }
        }
        None
    }

    pub fn display(&self) {
        println!("\nFIR Graph Adjacency List:");
        for node_idx in self.graph.graph.node_indices() {
            let node = &self.graph.graph[node_idx];
            let neighbors: Vec<String> = self.graph.graph.edges_directed(node_idx, petgraph::Direction::Outgoing)
                .map(|edge| self.graph.graph[edge.target()].name.as_ref().map(|n| n.to_string()).unwrap_or("<unnamed>".to_string()))
                .collect();
            let node_info = format!("{:?}", node.nt);
            println!("{} ({}): -> {:?}", node.name.as_ref().map(|n| n.to_string()).unwrap_or("<unnamed>".to_string()), node_info, neighbors);
        }
    }
}