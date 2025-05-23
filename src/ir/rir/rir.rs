use rusty_firrtl::*;
use indexmap::IndexMap;
use crate::ir::rir::rgraph::*;
use crate::ir::hierarchy::*;
use crate::common::RippleIRErr;
use crate::common::graphviz::*;

#[derive(Debug, Clone)]
pub struct RippleIR {
    pub name: Identifier,
    pub graphs: IndexMap<Identifier, RippleGraph>,
    pub hierarchy: Hierarchy,
}

impl RippleIR {
    pub fn new(name: Identifier) -> Self {
        Self {
            name,
            graphs: IndexMap::new(),
            hierarchy: Hierarchy::default(),
        }
    }

    pub fn export(&self, outdir: &str, pfx: &str) -> Result<(), RippleIRErr> {
        for (module, rg) in self.graphs.iter() {
            rg.export_graphviz(
                &format!("{}/{}-{}.{}.pdf", outdir, self.name.to_string(), module, pfx),
                None,
                None,
                false)?;
        }
        Ok(())
    }
}
