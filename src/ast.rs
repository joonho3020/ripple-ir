


#[derive(Debug, Clone)]
pub struct Circuit {
    pub identifier: String,
    pub modules: Vec<Module>
}

#[derive(Debug, Clone)]
pub struct Module {
    pub identifier: String,
// pub ports: Vec<Port>,
// pub stmts: Vec<Statements>
}

// #[derive(Debug, Clone)]
// pub struct Port {
// }
