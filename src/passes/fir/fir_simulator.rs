use crate::ir::fir::{FirGraph, FirNodeType};
use num_traits::ToPrimitive;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use rusty_firrtl::Int;

/// Simulator value for each node
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

pub struct FirSimulator {
    pub graph: FirGraph,
    pub values: HashMap<NodeIndex, FirValue>,
    pub next_values: HashMap<NodeIndex, FirValue>,
    pub out_values: HashMap<NodeIndex, FirValue>,
}

impl FirSimulator {
    /// Create simulator and split bundles
    pub fn new(graph: FirGraph) -> Self {
        let mut sim = Self { 
            graph, 
            values: HashMap::new(), 
            next_values: HashMap::new(),
            out_values: HashMap::new(),
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
                
                // Update values for non-register nodes immediately
                if !matches!(node.nt, FirNodeType::Reg) {
                    self.values.insert(idx, value.clone());
                }
                
                // Store values for register inputs (next cycle)
                for e in self.graph.graph.edges_directed(idx, petgraph::Direction::Outgoing) {
                    let tgt = e.target();
                    if let FirNodeType::Reg = self.graph.graph[tgt].nt {
                        self.next_values.insert(tgt, value.clone());
                    }
                }
            }
        }
        
        // Update registers: current output becomes previous, next value becomes current
        for idx in self.graph.graph.node_indices() {
            if let FirNodeType::Reg = self.graph.graph[idx].nt {
                // Store current output for this cycle
                let current_output = self.values.get(&idx).cloned().unwrap_or(FirValue::X);
                self.out_values.insert(idx, current_output);
                
                // Load next cycle's value
                if let Some(next_val) = self.next_values.get(&idx) {
                    self.values.insert(idx, next_val.clone());
                }
            }
        }
    }

    /// Set input value for bundle field (e.g., "io.a", "io.b")
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
        self.find_node_by_name(name).and_then(|idx| {
            match self.graph.graph[idx].nt {
                FirNodeType::Reg => self.out_values.get(&idx).cloned(),
                _ => self.values.get(&idx).cloned(),
            }
        })
    }

    /// Display graph adjacency list
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

    /// Display topological levels
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

    // --- Internal methods ---
    
    /// Find node by name, preferring Reg over Phi
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
            // Prefer Reg node, then Phi node
            matches.iter().find(|&&idx| matches!(self.graph.graph[idx].nt, crate::ir::fir::FirNodeType::Reg)).copied()
                .or_else(|| matches.iter().find(|&&idx| matches!(self.graph.graph[idx].nt, crate::ir::fir::FirNodeType::Phi(_))).copied())
                .or_else(|| Some(matches[0]))
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
        FirNodeType::Input
    }

    /// Calculate topological levels for simulation
    fn levels(&self) -> Vec<Vec<NodeIndex>> {
        let mut in_deg = HashMap::new();
        
        // Initialize in-degrees
        for idx in self.graph.graph.node_indices() {
            let node = &self.graph.graph[idx];
            if let FirNodeType::Reg = node.nt {
                in_deg.insert(idx, 0); // Registers always available
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

    /// Compute node value based on type and inputs
    fn compute_node_value(&self, idx: NodeIndex, node: &crate::ir::fir::FirNode) -> FirValue {
        use FirNodeType::*;
        match &node.nt {
            UIntLiteral(_, val) => FirValue::Int(val.clone()),
            Input => self.values.get(&idx).cloned().unwrap_or(FirValue::X),
            Reg => self.out_values.get(&idx).cloned().unwrap_or(FirValue::X),
            PrimOp2Expr(op) => self.compute_primop2_expr(idx, op),
            Output => self.get_input_value(idx),
            PrimOp1Expr1Int(op, param) => self.compute_primop1_expr1_int(idx, op, param),
            PrimOp1Expr2Int(op, hi, lo) => self.compute_primop1_expr2_int(idx, op, hi, lo),
            PrimOp1Expr(op) => self.compute_primop1_expr(idx, op),
            Phi(cond_path) => self.compute_phi_value(idx, cond_path),
            SMem(_) | CMem => FirValue::X,
            _ => FirValue::X,
        }
    }

    /// Get input value for node
    fn get_input_value(&self, idx: NodeIndex) -> FirValue {
        self.graph.graph.edges_directed(idx, petgraph::Direction::Incoming)
            .find_map(|e| self.values.get(&e.source()).cloned())
            .unwrap_or(FirValue::X)
    }

    /// Compute two-operand primitive operations
    fn compute_primop2_expr(&self, idx: NodeIndex, op: &rusty_firrtl::PrimOp2Expr) -> FirValue {
        let mut operand0 = None;
        let mut operand1 = None;
        
        // Collect operands by edge labels
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
            rusty_firrtl::PrimOp2Expr::Add => Int::from((a + b) as u32),
            rusty_firrtl::PrimOp2Expr::Sub => Int::from((a - b) as u32),
            rusty_firrtl::PrimOp2Expr::Mul => Int::from((a * b) as u32),
            rusty_firrtl::PrimOp2Expr::Div => {
                if b == 0 { return FirValue::X; } // Division by zero
                Int::from((a / b) as u32)
            }
            rusty_firrtl::PrimOp2Expr::Rem => {
                if b == 0 { return FirValue::X; } // Modulus by zero
                Int::from((a % b) as u32)
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
            rusty_firrtl::PrimOp2Expr::Eq => {
                if a == b { Int::from(1u32) } else { Int::from(0u32) }
            }
            rusty_firrtl::PrimOp2Expr::Neq => {
                if a != b { Int::from(1u32) } else { Int::from(0u32) }
            }
            rusty_firrtl::PrimOp2Expr::Dshl => {
                let shift_amount = b as u32;
                if shift_amount >= 32 {
                    Int::from(0u32)
                } else {
                    Int::from((a << shift_amount) as u32)
                }
            }
            rusty_firrtl::PrimOp2Expr::Dshr => {
                let shift_amount = b as u32;
                if shift_amount >= 32 {
                    Int::from(0u32)
                } else {
                    Int::from((a >> shift_amount) as u32)
                }
            }
            rusty_firrtl::PrimOp2Expr::And => Int::from((a & b) as u32),
            rusty_firrtl::PrimOp2Expr::Or => Int::from((a | b) as u32),
            rusty_firrtl::PrimOp2Expr::Xor => Int::from((a ^ b) as u32),
            rusty_firrtl::PrimOp2Expr::Cat => {
                let width_b = 32;
                let a_u64 = op0.0.to_u64().unwrap_or(0);
                let b_u64 = op1.0.to_u64().unwrap_or(0);
                let concat = ((a_u64 << width_b) | b_u64) & 0xFFFFFFFF;
                Int::from(concat as u32)
            },
            
            _ => return FirValue::X,
        };
        
        FirValue::Int(result)
    }

    /// Compute one-operand primitive operations with parameter
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
            rusty_firrtl::PrimOp1Expr1Int::Pad => {
                FirValue::Int(operand)
            }
            rusty_firrtl::PrimOp1Expr1Int::Shl => {
                let shift_amount = param.0.to_u32().unwrap_or(0);
                if shift_amount == 0 {
                    return FirValue::Int(operand);
                }
                
                let current_val = operand.0.to_u64().unwrap_or(0);
                let shifted_val = if shift_amount >= 32 {
                    0u64
                } else {
                    current_val << shift_amount
                };
                
                let result = shifted_val & 0xFFFFFFFF;
                FirValue::Int(Int::from(result as u32))
            }
            rusty_firrtl::PrimOp1Expr1Int::Shr => {
                let shift_amount = param.0.to_u32().unwrap_or(0);
                if shift_amount == 0 {
                    return FirValue::Int(operand);
                }
                
                let current_val = operand.0.to_u64().unwrap_or(0);
                let shifted_val = if shift_amount >= 32 {
                    0u64
                } else {
                    current_val >> shift_amount
                };
                
                let result = shifted_val & 0xFFFFFFFF;
                FirValue::Int(Int::from(result as u32))
            }
            rusty_firrtl::PrimOp1Expr1Int::Head => {
                let n = param.0.to_u32().unwrap_or(0);
                if n == 0 || n > 32 {
                    return FirValue::X;
                }
                let val = operand.0.to_u64().unwrap_or(0);
                let shift = 32 - n;
                let head_bits = (val >> shift) & ((1u64 << n) - 1);
                FirValue::Int(Int::from(head_bits as u32))
            }
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

    /// Compute one-operand primitive operations with two integer parameters
    fn compute_primop1_expr2_int(&self, idx: NodeIndex, op: &rusty_firrtl::PrimOp1Expr2Int, hi: &Int, lo: &Int) -> FirValue {
        let operand = self.graph.graph.edges_directed(idx, petgraph::Direction::Incoming)
            .filter_map(|e| {
                self.values.get(&e.source()).and_then(|v| {
                    if let FirValue::Int(i) = v { Some(i.clone()) } else { None }
                })
            })
            .next()
            .unwrap_or(Int::from(0u32));
        match op {
            rusty_firrtl::PrimOp1Expr2Int::BitSelRange => {
                let hi = hi.0.to_u32().unwrap_or(0);
                let lo = lo.0.to_u32().unwrap_or(0);
                if hi < lo || hi >= 32 || lo >= 32 {
                    return FirValue::X;
                }
                let val = operand.0.to_u64().unwrap_or(0);
                let width = hi - lo + 1;
                let mask = if width >= 32 { 0xFFFFFFFF } else { (1u64 << width) - 1 };
                let extracted = (val >> lo) & mask;
                FirValue::Int(Int::from(extracted as u32))
            }
            _ => FirValue::X,
        }
    }

    /// Compute one-operand primitive operations with no parameters
    fn compute_primop1_expr(&self, idx: NodeIndex, op: &rusty_firrtl::PrimOp1Expr) -> FirValue {
        let operand = self.graph.graph.edges_directed(idx, petgraph::Direction::Incoming)
            .filter_map(|e| {
                self.values.get(&e.source()).and_then(|v| {
                    if let FirValue::Int(i) = v { Some(i.clone()) } else { None }
                })
            })
            .next()
            .unwrap_or(Int::from(0u32));
        
        match op {
            rusty_firrtl::PrimOp1Expr::AsUInt => {
                FirValue::Int(operand)
            }
            rusty_firrtl::PrimOp1Expr::AsSInt => {
                FirValue::Int(operand)
            }
            rusty_firrtl::PrimOp1Expr::AsClock => {
                FirValue::Int(operand)
            }
            rusty_firrtl::PrimOp1Expr::Cvt => {
                // Convert unsigned to signed: if MSB is set, treat as negative
                let val = operand.0.to_u64().unwrap_or(0);
                if (val & 0x80000000) != 0 {
                    // MSB is set, convert to negative signed value
                    let signed_val = (0x100000000 - val) as i32;
                    FirValue::Int(Int::from((-signed_val) as u32))
                } else {
                    // MSB is clear, value remains the same
                    FirValue::Int(operand)
                }
            }
            rusty_firrtl::PrimOp1Expr::Neg => {
                // Negate the value (two's complement)
                let val = operand.0.to_u64().unwrap_or(0);
                let negated = if val == 0 {
                    0u32
                } else {
                    (0x100000000 - val) as u32
                };
                FirValue::Int(Int::from(negated))
            }
            rusty_firrtl::PrimOp1Expr::Not => {
                // Bitwise complement (NOT each bit)
                let val = operand.0.to_u64().unwrap_or(0);
                let complemented = (!val) & 0xFFFFFFFF; // Ensure 32-bit result
                FirValue::Int(Int::from(complemented as u32))
            }
            rusty_firrtl::PrimOp1Expr::Andr => {
                // AND reduction: true if all bits are 1, false otherwise
                let val = operand.0.to_u64().unwrap_or(0);
                let all_ones = (val & 0xFFFFFFFF) == 0xFFFFFFFF;
                FirValue::Int(Int::from(if all_ones { 1u32 } else { 0u32 }))
            }
            rusty_firrtl::PrimOp1Expr::Orr => {
                // OR reduction: true if any bit is 1, false otherwise
                let val = operand.0.to_u64().unwrap_or(0);
                let any_one = (val & 0xFFFFFFFF) != 0;
                FirValue::Int(Int::from(if any_one { 1u32 } else { 0u32 }))
            }
            rusty_firrtl::PrimOp1Expr::Xorr => {
                // XOR reduction: true if odd number of bits are 1, false otherwise
                let val = operand.0.to_u64().unwrap_or(0);
                let bit_count = (val & 0xFFFFFFFF).count_ones();
                let odd_ones = (bit_count % 2) == 1;
                FirValue::Int(Int::from(if odd_ones { 1u32 } else { 0u32 }))
            }
            _ => FirValue::X,
        }
    }

    /// Compute phi node value based on condition paths
    fn compute_phi_value(&self, idx: NodeIndex, _cond_path: &crate::ir::whentree::CondPathWithPrior) -> FirValue {
        // Get phi input edges with condition paths
        let mut phi_inputs: Vec<_> = self.graph.graph.edges_directed(idx, petgraph::Direction::Incoming)
            .filter_map(|e| {
                if let crate::ir::fir::FirEdgeType::PhiInput(input_cond_path, _flipped) = &e.weight().et {
                    let value = self.values.get(&e.source()).cloned();
                    let src_idx = e.source();
                    Some((input_cond_path.clone(), value, src_idx))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by priority, then by source index for determinism
        phi_inputs.sort_by(|(a, _, src_a), (b, _, src_b)| {
            let ord = a.cmp(b);
            if ord == std::cmp::Ordering::Equal {
                src_a.index().cmp(&src_b.index())
            } else {
                ord
            }
        });
        
        // Evaluate conditions and select value
        for (input_cond_path, value, _src_idx) in &phi_inputs {
            let condition_met = self.evaluate_condition_path(input_cond_path);
            if condition_met {
                return value.clone().unwrap_or(FirValue::X);
            }
        }
        
        // If no conditions met, return current register value
        for e in self.graph.graph.edges_directed(idx, petgraph::Direction::Outgoing) {
            if let crate::ir::fir::FirEdgeType::PhiOut = e.weight().et {
                let reg_idx = e.target();
                return self.values.get(&reg_idx).cloned().unwrap_or(FirValue::X);
            }
        }
        FirValue::X
    }

    /// Evaluate condition path
    fn evaluate_condition_path(&self, cond_path: &crate::ir::whentree::CondPathWithPrior) -> bool {
        for cond_with_prior in cond_path.iter() {
            if !self.evaluate_condition(&cond_with_prior.cond) {
                return false;
            }
        }
        true
    }

    /// Evaluate single condition
    fn evaluate_condition(&self, condition: &crate::ir::whentree::Condition) -> bool {
        use crate::ir::whentree::Condition;
        match condition {
            Condition::Root => true,
            Condition::When(expr) => self.evaluate_expr(expr),
            Condition::Else(expr) => !self.evaluate_expr(expr),
        }
    }

    /// Evaluate expression to boolean
    fn evaluate_expr(&self, expr: &rusty_firrtl::Expr) -> bool {
        match expr {
            rusty_firrtl::Expr::Reference(reference) => {
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
            _ => false,
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
            rusty_firrtl::Reference::RefIdxInt(_, _) | rusty_firrtl::Reference::RefIdxExpr(_, _) => None,
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

    /// Split bundle node into field nodes
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
            if src != bundle_idx && dst != bundle_idx {
                continue;
            }
            
            let edge_weight = edge.weight();
            let mut should_rewire = false;
            let mut new_src = src;
            let mut new_dst = dst;
            
            // Check edge metadata for bundle field references
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
        
        // Remove bundle node
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