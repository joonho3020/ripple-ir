use chirrtl_parser::ast::*;
use derivative::Derivative;
use petgraph::graph::{Graph, NodeIndex};
use indexmap::IndexMap;
use std::{u32, usize};
use std::hash::Hash;

use crate::define_index_type;

define_index_type!(BlockPrior);
define_index_type!(StmtPrior);

/// Represents the priority of an input edge going into a Phi node
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub struct PhiPrior {
    /// Priority between blocks
    /// - Smaller number means higher priority
    pub block: BlockPrior,

    /// Priority between statements within the same block
    /// - Smaller number means higher priority
    pub stmt: StmtPrior,
}

impl PhiPrior {
    pub fn new(block: BlockPrior, stmt: StmtPrior) -> Self {
        Self { block, stmt }
    }

    pub fn bottom() -> Self {
        Self { block: BlockPrior(u32::MIN), stmt: StmtPrior(u32::MIN) }
    }

    pub fn top() -> Self {
        Self { block: BlockPrior(u32::MAX-1), stmt: StmtPrior(u32::MAX-1) }
    }

    pub fn god() -> Self {
        Self { block: BlockPrior(u32::MAX), stmt: StmtPrior(u32::MAX) }
    }

    pub fn inc_block(&mut self) -> BlockPrior {
        self.block.0 += 1;
        self.block
    }
}

impl Ord for PhiPrior {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.block == other.block {
            self.stmt.cmp(&other.stmt)
        } else {
            self.block.cmp(&other.block)
        }
    }
}

#[derive(Debug, Default, Clone, Eq, Hash)]
pub struct CondWithPrior {
    /// Priority
    pub prior: PhiPrior,

    /// Condition
    pub cond: Condition,
}

impl CondWithPrior {
    pub fn new(prior: PhiPrior, cond: Condition) -> Self {
        Self { prior, cond }
    }
}

impl From<&WhenTreeNode> for CondWithPrior {
    fn from(value: &WhenTreeNode) -> Self {
        Self::new(value.prior, value.cond.clone())
    }
}

impl PartialEq for CondWithPrior {
    fn eq(&self, other: &Self) -> bool {
        self.cond == other.cond && self.prior == other.prior
    }
}

/// Higher prior               Lower priority
/// Lower value  <-----------> Higher value
/// Bottom                     Top
impl PartialOrd for CondWithPrior {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (&self.cond, &other.cond) {
            (Condition::When(se), Condition::Else(oe)) => {
                if se == oe {
                    if self.prior.stmt == other.prior.stmt {
                        // Matching when-else block
                        Some(std::cmp::Ordering::Greater)
                    } else {
                        // Compare between block won't work here as the priority
                        // value of the else is higher when:
                        // 1) the else is on top of the current when
                        // 2) the else is the corresponding case for the current when
                        Some(self.prior.stmt.cmp(&other.prior.stmt))
                    }
                } else {
                    // Different case, fall back to default comparison
                    Some(self.prior.cmp(&other.prior))
                }
            }
            (Condition::Else(se), Condition::When(oe)) => {
                if se == oe {
                    if self.prior.stmt == other.prior.stmt {
                        // Matching when-else block
                        Some(std::cmp::Ordering::Less)
                    } else {
                        // Similar to the above, just compare stmt
                        Some(self.prior.stmt.cmp(&other.prior.stmt))
                    }
                } else {
                    Some(self.prior.cmp(&other.prior))
                }
            }
            _ => {
                Some(self.prior.cmp(&other.prior))
            }
        }
    }
}

impl Ord for CondWithPrior {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct CondPath(pub Vec<CondWithPrior>);

impl CondPath {
    pub fn always_true(&self) -> bool {
        for cond in self.iter().map(|x| &x.cond) {
            match cond {
                Condition::When(..) |
                    Condition::Else(..) => {
                        return false;
                    }
                _ => {
                    continue;
                }
            }
        }
        return true;
    }

    pub fn get(&self, i: usize) -> &CondWithPrior {
        self.0.get(i).unwrap()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn top() -> Self {
        Self(vec![
            CondWithPrior::new(PhiPrior::top(), Condition::Root)
        ])
    }

    pub fn bottom() -> Self {
        Self(vec![
            CondWithPrior::new(PhiPrior::bottom(), Condition::Root)
        ])
    }

    pub fn iter(&self) -> std::slice::Iter<'_, CondWithPrior> {
        self.0.iter()
    }

    pub fn from_vec(val: Vec<CondWithPrior>) -> Self {
        Self { 0: val }
    }

    pub fn push(&mut self, pcond: CondWithPrior) {
        self.0.push(pcond)
    }

    pub fn last_mut(&mut self) -> Option<&mut CondWithPrior> {
        self.0.last_mut()
    }

    pub fn last(&self) -> Option<&CondWithPrior> {
        self.0.last()
    }

    /// Collect all the selection `Expr` in this condition chain
    pub fn collect_sels(&self) -> Vec<Expr> {
        let mut ret = vec![];
        for cond in self.iter().map(|x| &x.cond) {
            match cond {
                Condition::When(e) |
                Condition::Else(e) => {
                    ret.push(e.clone())
                }
                _ => { }
            }
        }
        return ret;
    }

    pub fn contains(&self, other: &Self) -> bool {
        if self.len() > other.len() {
            return false;
        }
        let other_len = other.len();
        for (i, other_pcond) in other.iter().enumerate() {
            if i == other_len - 1 {
                assert!(other_pcond.cond == Condition::Root);
            } else {
                if self.0[i] != *other_pcond {
                    return false;
                }
            }
        }
        true
    }
}

impl PartialOrd for CondPath {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let len = self.0.len().min(other.0.len());
        for i in 0..len {
            let self_pcond = &self.0[i];
            let other_pcond = &other.0[i];
            if self.0[i] == other.0[i] {
                continue;
            } else {
                return Some(self_pcond.cmp(&other_pcond));
            }
        }

        if self.0.len() > other.0.len() {
            Some(std::cmp::Ordering::Greater)
        } else {
            Some(std::cmp::Ordering::Less)
        }
    }
}

impl Ord for CondPath {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// When/Else conditions
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Condition {
    /// Root condition (basically always executed)
    #[default]
    Root,

    /// Condition is true when Box<Condition> && Expr
    When(Expr),

    /// Condition is true when Box<Condition> && !Expr
    Else(Expr),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StmtWithPrior {
    pub stmt: Stmt,
    pub prior: Option<StmtPrior>,
}

impl StmtWithPrior {
    pub fn new(stmt: Stmt, prior: Option<StmtPrior>) -> Self {
        Self { stmt, prior }
    }
}

impl PartialOrd for StmtWithPrior {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self.prior, other.prior) {
            (Some(sp), Some(op)) => {
                Some(sp.cmp(&op))
            }
            (Some(_), None) => {
                Some(std::cmp::Ordering::Less)
            }
            (None, Some(_)) => {
                Some(std::cmp::Ordering::Greater)
            }
            (None, None) => {
                Some(std::cmp::Ordering::Less)
            }
        }
    }
}

impl Ord for StmtWithPrior {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(&other).unwrap()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct CondPathWithPrior {
    /// Path to the whentree leaf
    pub path: CondPath,

    /// Stmt priority within the leaf
    pub prior: StmtPrior
}

impl CondPathWithPrior {
    pub fn new(path: CondPath, prior: StmtPrior) -> Self {
        Self { path, prior }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, CondWithPrior> {
        self.path.iter()
    }
}

impl PartialOrd for CondPathWithPrior {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.path == other.path {
            Some(self.prior.cmp(&other.prior))
        } else {
            Some(self.path.cmp(&other.path))
        }
    }
}

impl Ord for CondPathWithPrior {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Derivative, Default, Clone, Eq)]
#[derivative(Debug)]
pub struct WhenTreeNode {
    /// Condition to reach this node
    pub cond: Condition,

    /// Priority of this node
    /// - prior.block: Block level priority
    /// - prior.stmt: Stmt level priority, useful when there are multiple
    /// whens with the same condition
    pub prior: PhiPrior,

    /// Statements in this tree node
    #[derivative(Debug="ignore")]
    pub stmts: Vec<StmtWithPrior>
}

impl WhenTreeNode {
    pub fn new(cond: Condition, priority: PhiPrior) -> Self {
        Self { cond, prior: priority, stmts: vec![] }
    }

    pub fn god() -> Self {
        Self::new(Condition::Root, PhiPrior::god())
    }

    pub fn top() -> Self {
        Self::new(Condition::Root, PhiPrior::top())
    }

    pub fn bottom() -> Self {
        Self::new(Condition::Root, PhiPrior::bottom())
    }
}

impl Hash for WhenTreeNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.cond.hash(state);
        self.prior.hash(state);
    }
}

impl PartialEq for WhenTreeNode {
    fn eq(&self, other: &Self) -> bool {
        self.cond == other.cond && self.prior == other.prior
    }
}

/// Map from WhenTreeLeafNodes to their condition path
pub type LeafToPath<'a> = IndexMap<&'a WhenTreeNode, CondPath>;

pub type WhenTreeEdge = ();

pub type WhenTreeGraph = Graph<WhenTreeNode, WhenTreeEdge>;

/// Represents a tree of decision blocks
///
/// ```
/// stmt_5
/// when (a)
///   stmt_0
///   when (b)
///     stmt_1
///   else
///     stmt_2
/// else
///   stmt_3
/// stmt_4
/// ```
///
/// The above when statements will produce a tree like this:
///
/// ```
/// God
/// |- stmt_4 (highest priority)
/// |- when (a)
/// |  |- when (b)
/// |  |  |- stmt_1
/// |  |- else
/// |  |  |- stmt_2
/// |  |stmt_0
/// |- else
/// |  |- stmt_3
/// |- stmt_5 (lowest priority)
/// ```
#[derive(Debug, Clone, Default)]
pub struct WhenTree {
    pub graph: WhenTreeGraph,
    pub god: Option<NodeIndex>
}

#[cfg(test)]
mod test {
    use super::*;
    use chirrtl_parser::parse_circuit;
    use crate::common::RippleIRErr;

    fn run(path: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(path)?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for cm in circuit.modules {
            match cm.as_ref() {
                CircuitModule::Module(m) => {
                    let whentree = WhenTree::build_from_stmts(m.stmts.as_ref());
                    whentree.print_tree();
                }
                _ => { }
            }
        }
        Ok(())
    }

    #[test]
    fn gcd() {
        run("./test-inputs/GCD.fir").expect("gcd");
    }

    #[test]
    fn nested_when() {
        run("./test-inputs/NestedWhen.fir").expect("nested_when");
    }
}
