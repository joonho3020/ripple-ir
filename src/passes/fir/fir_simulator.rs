use crate::ir::fir::{FirGraph, FirNodeType};
use num_traits::ToPrimitive;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use rusty_firrtl::Int;

/// Value stored in the simulator for each node
#[derive(Debug, Clone, PartialEq)]
pub enum FirValue {
    X,
    Int(Int),
}

impl FirValue {
    pub fn to_int(&self) -> Option<Int> {
        if let FirValue::Int(i) = self { Some(i.clone()) } else { None }
    }
}

/// Main FIRRTL simulator
pub struct FirSimulator {
    pub graph: FirGraph,
    pub values: HashMap<NodeIndex, FirValue>,
    pub next_values: HashMap<NodeIndex, FirValue>,
}

impl FirSimulator {
    /// Create a new simulator and split bundles
    pub fn new(graph: FirGraph) -> Self {
        let mut sim = Self { 
            graph, 
            values: HashMap::new(), 
            next_values: HashMap::new() 
        };
        sim.split_bundles();
        sim
    }

    /// Run one simulation cycle (level by level)
    pub fn run(&mut self) {
        self.next_values.clear();
        
        // Process nodes level by level
        for level in self.levels() {
            for &idx in &level {
                let node = &self.graph.graph[idx];
                let value = self.compute_node_value(idx, node);
                
                // Store value for this node
                self.values.insert(idx, value.clone());
                
                // Handle edges to registers - store values for next cycle
                for e in self.graph.graph.edges_directed(idx, petgraph::Direction::Outgoing) {
                    let tgt = e.target();
                    let edge_type = &e.weight().et;
                    
                    match edge_type {
                        crate::ir::fir::FirEdgeType::PhiOut => {
                            // Phi node output - update the target register for next cycle
                            self.next_values.insert(tgt, value.clone());
                        }
                        _ => {
                            // Regular edge - if target is a register, store for next cycle
                            if let FirNodeType::Reg = self.graph.graph[tgt].nt {
                                self.next_values.insert(tgt, value.clone());
                            }
                        }
                    }
                }
            }
        }
        
        // Update registers with their next values for the next cycle
        for idx in self.graph.graph.node_indices() {
            if let FirNodeType::Reg = self.graph.graph[idx].nt {
                if let Some(val) = self.next_values.get(&idx) {
                    self.values.insert(idx, val.clone());
                }
            }
        }
    }

    /// Set input value for a bundle field (e.g., "io.a", "io.b")
    pub fn set_bundle_input(&mut self, bundle_name: &str, field_name: &str, value: Int) {
        let field_node_name = format!("{}.{}", bundle_name, field_name);
        
        // Check if field node already exists
        if let Some(idx) = self.find_node_by_name(&field_node_name) {
            self.values.insert(idx, FirValue::Int(value));
            return;
        }
        
        // Create new field node with correct type from TypeTree
        let node_type = self.get_field_type(bundle_name, field_name);
        let field_node = crate::ir::fir::FirNode::new(
            Some(rusty_firrtl::Identifier::Name(field_node_name)),
            node_type,
            None
        );
        let idx = self.graph.graph.add_node(field_node);
        self.values.insert(idx, FirValue::Int(value));
    }

    /// Set input value for a node by name
    pub fn set_input(&mut self, name: &str, value: Int) {
        if let Some(idx) = self.find_node_by_name(name) {
            self.values.insert(idx, FirValue::Int(value));
        }
    }

    /// Get the output value for a node by name
    pub fn get_output(&self, name: &str) -> Option<FirValue> {
        self.find_node_by_name(name).and_then(|idx| self.values.get(&idx).cloned())
    }

    /// Display the adjacency list of the FIR graph
    pub fn display(&self) {
        println!("\nFIR Graph Adjacency List:");
        for idx in self.graph.graph.node_indices() {
            let node = &self.graph.graph[idx];
            let neighbors: Vec<_> = self.graph.graph.edges_directed(idx, petgraph::Direction::Outgoing)
                .map(|e| self.graph.graph[e.target()].name.as_ref().map(|n| n.to_string()).unwrap_or("<unnamed>".to_string()))
                .collect();
            let name = node.name.as_ref().map(|n| n.to_string()).unwrap_or("<unnamed>".to_string());
            println!("{} ({:?}): -> {:?}", name, node.nt, neighbors);
        }
    }

    /// Display the levelization (topological levels) of the FIR graph
    pub fn display_levelization(&self) {
        println!("\nFIR Graph Levelization:");
        for (i, level) in self.levels().iter().enumerate() {
            println!("Level {}: [", i);
            for &idx in level {
                let node = &self.graph.graph[idx];
                let name = node.name.as_ref().map(|n| n.to_string()).unwrap_or_else(|| format!("<unnamed_{}>", idx.index()));
                println!("  {} ({:?})", name, node.nt);
            }
            println!("]");
        }
    }

    // --- Internal helper methods ---

    /// Find node by name, preferring Phi node if multiple exist
    pub fn find_node_by_name(&self, name: &str) -> Option<NodeIndex> {
        let matches = self.graph.graph.node_indices()
            .filter(|&idx| {
                self.graph.graph[idx].name.as_ref()
                    .map(|n| n.to_string() == name)
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        if matches.is_empty() {
            None
        } else if matches.len() == 1 {
            Some(matches[0])
        } else {
            // Prefer Phi node if present
            matches.iter().find(|&&idx| matches!(self.graph.graph[idx].nt, crate::ir::fir::FirNodeType::Phi(_))).copied().or_else(|| Some(matches[0]))
        }
    }

    /// Get field type from TypeTree
    fn get_field_type(&self, bundle_name: &str, field_name: &str) -> FirNodeType {
        for idx in self.graph.graph.node_indices() {
            let node = &self.graph.graph[idx];
            if node.name.as_ref().map(|n| n.to_string()) == Some(bundle_name.to_string()) {
                if let Some(typetree) = &node.ttree {
                    if let Some(view) = typetree.view() {
                        let field_ref = rusty_firrtl::Reference::RefDot(
                            Box::new(rusty_firrtl::Reference::Ref(rusty_firrtl::Identifier::Name(bundle_name.to_string()))),
                            rusty_firrtl::Identifier::Name(field_name.to_string())
                        );
                        
                        if let Some(field_id) = view.subtree_root(&field_ref) {
                            if let Some(field_node_info) = view.get_node(field_id) {
                                return match field_node_info.dir {
                                    crate::ir::typetree::tnode::TypeDirection::Outgoing => FirNodeType::Input,
                                    crate::ir::typetree::tnode::TypeDirection::Incoming => FirNodeType::Output,
                                };
                            }
                        }
                    }
                }
                break;
            }
        }
        FirNodeType::Input // Default fallback
    }

    /// Calculate topological levels for simulation
    fn levels(&self) -> Vec<Vec<NodeIndex>> {
        let mut in_deg = HashMap::new();
        
        // Initialize in-degrees
        for idx in self.graph.graph.node_indices() {
            let node = &self.graph.graph[idx];
            // For registers, set in-degree to 0 (they're always available)
            if let FirNodeType::Reg = node.nt {
                in_deg.insert(idx, 0);
            } else {
                // Count incoming edges, excluding clock and reset
                let deg = self.graph.graph.edges_directed(idx, petgraph::Direction::Incoming)
                    .filter(|e| !matches!(e.weight().et, crate::ir::fir::FirEdgeType::Clock | crate::ir::fir::FirEdgeType::Reset))
                    .count();
                in_deg.insert(idx, deg);
            }
        }
        
        // Build levels
        let mut levels = Vec::new();
        let mut remaining: Vec<_> = self.graph.graph.node_indices().collect();
        
        while !remaining.is_empty() {
            let mut this_level = Vec::new();
            let mut next = Vec::new();
            
            for &idx in &remaining {
                if in_deg[&idx] == 0 {
                    this_level.push(idx);
                } else {
                    next.push(idx);
                }
            }
            
            if this_level.is_empty() { break; }
            
            // Update in-degrees for next level
            for &idx in &this_level {
                for e in self.graph.graph.edges_directed(idx, petgraph::Direction::Outgoing) {
                    let tgt = e.target();
                    if !matches!(e.weight().et, crate::ir::fir::FirEdgeType::Clock | crate::ir::fir::FirEdgeType::Reset) {
                        if let Some(deg) = in_deg.get_mut(&tgt) {
                            if *deg > 0 {
                                *deg -= 1;
                            }
                        }
                    }
                }
            }
            
            levels.push(this_level);
            remaining = next;
        }
        
        levels
    }

    /// Compute the value for a node based on its type and inputs
    fn compute_node_value(&self, idx: NodeIndex, node: &crate::ir::fir::FirNode) -> FirValue {
        use FirNodeType::*;
        match &node.nt {
            UIntLiteral(_, val) => FirValue::Int(val.clone()),
            Input => self.values.get(&idx).cloned().unwrap_or(FirValue::X),
            Reg => self.values.get(&idx).cloned().unwrap_or(FirValue::X),
            PrimOp2Expr(op) => self.compute_primop2_expr(idx, op),
            Output => self.get_input_value(idx),
            PrimOp1Expr1Int(op, param) => self.compute_primop1_expr1_int(idx, op, param),
            Phi(cond_path) => self.compute_phi_value(idx, cond_path),
            SMem(_) | CMem => FirValue::X,
            _ => FirValue::X,
        }
    }

    /// Get input value for a node
    fn get_input_value(&self, idx: NodeIndex) -> FirValue {
        self.graph.graph.edges_directed(idx, petgraph::Direction::Incoming)
            .find_map(|e| self.values.get(&e.source()).cloned())
            .unwrap_or(FirValue::X)
    }

    /// Compute two-operand primitive operations
    fn compute_primop2_expr(&self, idx: NodeIndex, op: &rusty_firrtl::PrimOp2Expr) -> FirValue {
        // Collect operands in the correct order based on edge labels
        let mut operand0 = None;
        let mut operand1 = None;
        
        for e in self.graph.graph.edges_directed(idx, petgraph::Direction::Incoming) {
            if let Some(value) = self.values.get(&e.source()) {
                if let FirValue::Int(int_val) = value {
                    match e.weight().et {
                        crate::ir::fir::FirEdgeType::Operand0 => operand0 = Some(int_val.clone()),
                        crate::ir::fir::FirEdgeType::Operand1 => operand1 = Some(int_val.clone()),
                        _ => {}
                    }
                }
            }
        }
        
        if operand0.is_none() || operand1.is_none() {
            return FirValue::X;
        }
        
        let op0 = operand0.unwrap();
        let op1 = operand1.unwrap();
        let a = op0.0.to_i64().unwrap();
        let b = op1.0.to_i64().unwrap();
        
        let result = match op {
            rusty_firrtl::PrimOp2Expr::And => Int::from((a & b) as u32),
            rusty_firrtl::PrimOp2Expr::Or => Int::from((a | b) as u32),
            rusty_firrtl::PrimOp2Expr::Eq => {
                if a == b { Int::from(1u32) } else { Int::from(0u32) }
            }
            rusty_firrtl::PrimOp2Expr::Neq => {
                if a != b { Int::from(1u32) } else { Int::from(0u32) }
            }
            rusty_firrtl::PrimOp2Expr::Lt => {
                if a < b { Int::from(1u32) } else { Int::from(0u32) }
            }
            rusty_firrtl::PrimOp2Expr::Leq => {
                if a <= b { Int::from(1u32) } else { Int::from(0u32) }
            }
            rusty_firrtl::PrimOp2Expr::Gt => {
                if a > b { Int::from(1u32) } else { Int::from(0u32) }
            }
            rusty_firrtl::PrimOp2Expr::Geq => {
                if a >= b { Int::from(1u32) } else { Int::from(0u32) }
            }
            rusty_firrtl::PrimOp2Expr::Add => Int::from((a + b) as u32),
            rusty_firrtl::PrimOp2Expr::Sub => Int::from((a - b) as u32),
            _ => return FirValue::X,
        };
        
        FirValue::Int(result)
    }

    /// Compute one-operand primitive operations with one integer parameter
    fn compute_primop1_expr1_int(&self, idx: NodeIndex, op: &rusty_firrtl::PrimOp1Expr1Int, param: &Int) -> FirValue {
        let operand = self.graph.graph.edges_directed(idx, petgraph::Direction::Incoming)
            .filter_map(|e| {
                self.values.get(&e.source()).and_then(|v| {
                    if let FirValue::Int(i) = v { Some(i.clone()) } else { None }
                })
            })
            .next()
            .unwrap_or(Int::from(0u32));
        
        match op {
            rusty_firrtl::PrimOp1Expr1Int::Tail => {
                let bits = param.0.to_u32().unwrap_or(0);
                if bits == 0 {
                    FirValue::Int(operand)
                } else {
                    let val = operand.0.to_u64().unwrap_or(0);
                    let mask = (1u64 << (32 - bits)) - 1;
                    FirValue::Int(Int::from((val & mask) as u32))
                }
            }
            _ => FirValue::X,
        }
    }

    /// Compute phi node value as a chain of muxes based on condition paths
    fn compute_phi_value(&self, idx: NodeIndex, _cond_path: &crate::ir::whentree::CondPathWithPrior) -> FirValue {
        // Get all phi input edges with their condition paths
        let mut phi_inputs: Vec<_> = self.graph.graph.edges_directed(idx, petgraph::Direction::Incoming)
            .filter_map(|e| {
                if let crate::ir::fir::FirEdgeType::PhiInput(input_cond_path, _flipped) = &e.weight().et {
                    let value = self.values.get(&e.source()).cloned();
                    let src_idx = e.source();
                    let src_name = self.graph.graph[src_idx].name.as_ref().map(|n| n.to_string()).unwrap_or_else(|| format!("node_{}", src_idx.index()));
                    Some((input_cond_path.clone(), value, src_idx, src_name, e.weight().et.clone()))
                } else {
                    None
                }
            })
            .collect();
        // Sort by priority (lower priority = higher precedence), then by source node index for determinism
        phi_inputs.sort_by(|(a, _, src_a, _, _), (b, _, src_b, _, _)| {
            let ord = a.cmp(b);
            if ord == std::cmp::Ordering::Equal {
                src_a.index().cmp(&src_b.index())
            } else {
                ord
            }
        });
        
        // Evaluate conditions and select the appropriate value
        for (input_cond_path, value, src_idx, src_name, edge_type) in &phi_inputs {
            let condition_met = self.evaluate_condition_path(input_cond_path);
            if condition_met {
                return value.clone().unwrap_or(FirValue::X);
            }
        }
        // If no conditions are met, return the current value of the register this phi node feeds
        let mut reg_val = None;
        for e in self.graph.graph.edges_directed(idx, petgraph::Direction::Outgoing) {
            if let crate::ir::fir::FirEdgeType::PhiOut = e.weight().et {
                let reg_idx = e.target();
                reg_val = self.values.get(&reg_idx).cloned();
                break;
            }
        }
        reg_val.unwrap_or(FirValue::X)
    }

    /// Evaluate a condition path to determine if it's true
    fn evaluate_condition_path(&self, cond_path: &crate::ir::whentree::CondPathWithPrior) -> bool {
        for cond_with_prior in cond_path.iter() {
            if !self.evaluate_condition(&cond_with_prior.cond) {
                return false;
            }
        }
        true
    }

    /// Evaluate a single condition
    fn evaluate_condition(&self, condition: &crate::ir::whentree::Condition) -> bool {
        use crate::ir::whentree::Condition;
        match condition {
            Condition::Root => true,
            Condition::When(expr) => self.evaluate_expr(expr),
            Condition::Else(expr) => !self.evaluate_expr(expr),
        }
    }

    /// Evaluate an expression to a boolean value
    fn evaluate_expr(&self, expr: &rusty_firrtl::Expr) -> bool {
        match expr {
            rusty_firrtl::Expr::Reference(reference) => {
                // Find the node by reference and get its value
                if let Some(idx) = self.find_node_by_reference(reference) {
                    if let Some(value) = self.values.get(&idx) {
                        if let FirValue::Int(int_val) = value {
                            return int_val.0.to_u64().unwrap_or(0) != 0;
                        }
                    }
                }
                false
            }
            rusty_firrtl::Expr::UIntInit(_, val) => val.0.to_u64().unwrap_or(0) != 0,
            rusty_firrtl::Expr::SIntInit(_, val) => val.0.to_i64().unwrap_or(0) != 0,
            _ => false, // For now, only handle simple expressions
        }
    }

    /// Find node by reference
    fn find_node_by_reference(&self, reference: &rusty_firrtl::Reference) -> Option<NodeIndex> {
        match reference {
            rusty_firrtl::Reference::Ref(identifier) => {
                self.find_node_by_name(&identifier.to_string())
            }
            rusty_firrtl::Reference::RefDot(parent, field) => {
                let field_name = format!("{}.{}", parent.to_string(), field.to_string());
                self.find_node_by_name(&field_name)
            }
            rusty_firrtl::Reference::RefIdxInt(_, _) => None, // Handle array indexing later if needed
            rusty_firrtl::Reference::RefIdxExpr(_, _) => None, // Handle array indexing later if needed
        }
    }

    /// Detect and split all bundle nodes based on their TypeTree
    fn split_bundles(&mut self) {
        let bundle_nodes: Vec<_> = self.graph.graph.node_indices()
            .filter(|&idx| {
                let node = &self.graph.graph[idx];
                if let Some(typetree) = &node.ttree {
                    if let Some(view) = typetree.view() {
                        if let Some(root_node) = view.root_node() {
                            matches!(root_node.tpe, crate::ir::typetree::tnode::TypeTreeNodeType::Fields)
                        } else { false }
                    } else { false }
                } else { false }
            })
            .collect();
        
        for bundle_idx in bundle_nodes {
            let bundle_name = self.graph.graph[bundle_idx].name.as_ref()
                .map(|n| n.to_string())
                .unwrap_or_else(|| format!("bundle_{}", bundle_idx.index()));
            self.split_bundle(&bundle_name);
        }
    }

    /// Split a bundle node into individual field nodes
    fn split_bundle(&mut self, bundle_name: &str) {
        use rusty_firrtl::Identifier;
        use crate::ir::fir::FirNode;

        // Find bundle nodes
        let bundle_nodes: Vec<_> = self.graph.graph.node_indices()
            .filter(|&idx| self.graph.graph[idx].name.as_ref().map(|n| n.to_string()) == Some(bundle_name.to_string()))
            .collect();
        
        if bundle_nodes.is_empty() { return; }
        
        // Get first bundle node with TypeTree
        let bundle_idx = bundle_nodes.iter()
            .find(|&&idx| self.graph.graph[idx].ttree.is_some())
            .copied()
            .unwrap_or(bundle_nodes[0]);
        
        // Extract field information from TypeTree
        let mut field_nodes = HashMap::new();
        if let Some(typetree) = &self.graph.graph[bundle_idx].ttree {
            if let Some(view) = typetree.view() {
                let all_refs = view.all_possible_references(self.graph.graph[bundle_idx].name.as_ref().unwrap().clone());
                
                for reference in all_refs {
                    if let rusty_firrtl::Reference::RefDot(parent, field_name) = &reference {
                        if parent.to_string() == bundle_name {
                            let field_node_name = format!("{}.{}", bundle_name, field_name.to_string());
                            let node_type = if let Some(field_id) = view.subtree_root(&reference) {
                                if let Some(field_node_info) = view.get_node(field_id) {
                                    match field_node_info.dir {
                                        crate::ir::typetree::tnode::TypeDirection::Outgoing => FirNodeType::Input,
                                        crate::ir::typetree::tnode::TypeDirection::Incoming => FirNodeType::Output,
                                    }
                                } else { FirNodeType::Input }
                            } else { FirNodeType::Input };
                            
                            field_nodes.insert(field_name.to_string(), (field_node_name, node_type));
                        }
                    }
                }
            }
        }

        // Create field nodes
        let mut field_indices = HashMap::new();
        let mut field_names: Vec<_> = field_nodes.keys().collect();
        field_names.sort();
        for field_name in field_names {
            let (field_node_name, node_type) = &field_nodes[field_name];
            let existing = self.graph.graph.node_indices().find(|&idx| {
                self.graph.graph[idx].name.as_ref().map(|n| n.to_string()) == Some(field_node_name.clone())
            });
            
            let field_idx = if let Some(idx) = existing {
                idx
            } else {
                let node = FirNode::new(Some(Identifier::Name(field_node_name.clone())), node_type.clone(), None);
                self.graph.graph.add_node(node)
            };
            
            field_indices.insert(field_name.clone(), field_idx);
        }

        // Rewire edges
        let mut edges_to_add = vec![];
        let mut edges_to_remove = vec![];
        for edge in self.graph.graph.edge_references() {
            let (src, dst) = (edge.source(), edge.target());
            // Only consider edges where src or dst is the current bundle node
            if src != bundle_idx && dst != bundle_idx {
                continue; // skip edges not involving the bundle being split
            }
            let edge_weight = edge.weight();
            let mut should_rewire = false;
            let mut new_src = src;
            let mut new_dst = dst;
            
            // Check if edge metadata src references a bundle field
            if let rusty_firrtl::Expr::Reference(reference) = &edge_weight.src {
                if let rusty_firrtl::Reference::RefDot(parent, field) = reference {
                    if parent.to_string() == bundle_name {
                        if let Some(field_idx) = field_indices.get(&field.to_string()) {
                            new_src = *field_idx;
                            should_rewire = true;
                        }
                    }
                }
            }
            // Check if edge metadata dst references a bundle field
            if let Some(rusty_firrtl::Reference::RefDot(parent, field)) = &edge_weight.dst {
                if parent.to_string() == bundle_name {
                    if let Some(field_idx) = field_indices.get(&field.to_string()) {
                        new_dst = *field_idx;
                        should_rewire = true;
                    }
                }
            }
            if should_rewire {
                edges_to_add.push((new_src, new_dst, edge_weight.clone()));
                edges_to_remove.push(edge.id());
            }
        }
        
        // Apply edge changes
        for edge_id in edges_to_remove {
            self.graph.graph.remove_edge(edge_id);
        }
        for (src, dst, weight) in edges_to_add {
            self.graph.graph.add_edge(src, dst, weight);
        }
        
        // Remove the bundle node and its edges
        // Only remove edges that are still attached to the bundle node
        let mut edges_to_remove = vec![];
        for edge in self.graph.graph.edges(bundle_idx).collect::<Vec<_>>() {
            edges_to_remove.push((edge.source(), edge.target()));
        }
        for (src, dst) in edges_to_remove {
            self.graph.graph.remove_edge(self.graph.graph.find_edge(src, dst).unwrap());
        }
        self.graph.graph.remove_node(bundle_idx);

    }
}