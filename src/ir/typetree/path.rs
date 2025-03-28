use std::hash::{Hash, Hasher};
use chirrtl_parser::ast::*;
use crate::ir::typetree::tnode::*;

/// Used for representing a path in the `TypeTree`
#[derive(Debug, Clone, Eq)]
pub struct TypeTreeNodePath {
    /// Direction of the type
    dir: TypeDirection,

    /// Type of this node
    tpe: TypeTreeNodeType,

    /// Reference representing the path in the `TypeTree`
    rc: Option<Reference>,
}

impl Hash for TypeTreeNodePath {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tpe.hash(state);
        self.rc.hash(state);
    }
}

impl PartialEq for TypeTreeNodePath {
    fn eq(&self, other: &Self) -> bool {
        self.rc == other.rc
    }
}

impl TypeTreeNodePath {
    pub fn new(dir: TypeDirection, tpe: TypeTreeNodeType, rc: Option<Reference>) -> Self {
        Self { dir, tpe, rc }
    }

    /// Add a `child` node to the path in the `TypeTree`
    pub fn append(&mut self, child: &TypeTreeNode) {
        self.dir = child.dir;
        self.rc = match &self.rc {
            None => Some(Reference::Ref(child.name.clone().unwrap())),
            Some(par) => {
                match self.tpe {
                    TypeTreeNodeType::Ground(_) => Some(Reference::Ref(child.name.clone().unwrap())),
                    TypeTreeNodeType::Fields    => Some(Reference::RefDot(Box::new(par.clone()), child.name.clone().unwrap())),
                    TypeTreeNodeType::Array     => {
                        let id = match &child.name {
                            Some(Identifier::ID(i)) => i,
                            _ => panic!("Array type should have ID as child, got {:?}", child.name)
                        };
                        Some(Reference::RefIdxInt(Box::new(par.clone()), id.clone()))
                    }
                }
            }
        };
        self.tpe = child.tpe.clone();
    }
}
