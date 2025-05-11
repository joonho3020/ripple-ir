use std::collections::{HashMap, HashSet};
use std::borrow::Cow;
use crate::ast::*;

/// Represents a target for inlining, similar to FIRRTL's InlineAnnotation
pub enum InlineTarget {
    /// Inline all instances of the specified module
    Module(Identifier),
    /// Inline a specific instance in a module
    Instance(Identifier, Identifier), // (module_name, instance_name)
}

/// Configuration for inlining
pub struct InlineConfig {
    /// Modules to inline (all instances of these modules will be inlined)
    pub modules: HashSet<Identifier>,
    /// Specific instances to inline (module_name -> set of instance_names)
    pub instances: HashMap<Identifier, HashSet<Identifier>>,
}

/// Perform inlining transformation on a Circuit
pub fn inline(circuit: Circuit, targets: Vec<InlineTarget>) -> Circuit {
    // Collect modules and instances to inline
    let mut config = InlineConfig {
        modules: HashSet::new(),
        instances: HashMap::new(),
    };
    
    for target in targets {
        match target {
            InlineTarget::Module(module_name) => {
                config.modules.insert(module_name);
            },
            InlineTarget::Instance(module_name, instance_name) => {
                config.instances
                    .entry(module_name)
                    .or_insert_with(HashSet::new)
                    .insert(instance_name);
            }
        }
    }
    
    // Build module lookup map
    let module_map = build_module_map(&circuit);
    
    // Validate targets
    validate_inline_targets(&circuit, &config, &module_map);
    
    // Perform inlining
    let inlined_modules = circuit.modules
        .into_iter()
        .map(|module_box| {
            let module = *module_box;
            match module {
                CircuitModule::Module(m) => {
                    let inlined_module = inline_module(m, &config, &module_map);
                    Box::new(CircuitModule::Module(inlined_module))
                },
                _ => Box::new(module),
            }
        })
        .collect();
    
    Circuit {
        modules: inlined_modules,
        ..circuit
    }
}

/// Build a map from module name to module definition
fn build_module_map(circuit: &Circuit) -> HashMap<Identifier, CircuitModule> {
    let mut map = HashMap::new();
    for module_box in &circuit.modules {
        let module = &**module_box;
        let name = match module {
            CircuitModule::Module(m) => m.name.clone(),
            CircuitModule::ExtModule(m) => m.name.clone(),
        };
        map.insert(name, module.clone());
    }
    map
}

/// Validate that inlining targets exist and are not external modules
fn validate_inline_targets(
    circuit: &Circuit,
    config: &InlineConfig, 
    module_map: &HashMap<Identifier, CircuitModule>
) {
    // Check that all modules to inline exist and are not external
    for module_name in &config.modules {
        if !module_map.contains_key(module_name) {
            panic!("Module to inline does not exist: {:?}", module_name);
        }
        match module_map.get(module_name).unwrap() {
            CircuitModule::ExtModule(_) => {
                panic!("Cannot inline external module: {:?}", module_name);
            },
            _ => {}
        }
    }
    
    // Check that all instances to inline exist and their modules are not external
    for (module_name, instances) in &config.instances {
        match module_map.get(module_name) {
            Some(CircuitModule::Module(m)) => {
                // Check each instance exists
                for instance_name in instances {
                    let instance_exists = m.stmts.iter().any(|stmt| {
                        if let Stmt::Inst(name, _, _) = &**stmt {
                            name == instance_name
                        } else {
                            false
                        }
                    });
                    
                    if !instance_exists {
                        panic!("Instance to inline doesn't exist: {:?} in module {:?}", 
                               instance_name, module_name);
                    }
                }
            },
            Some(CircuitModule::ExtModule(_)) => {
                panic!("Cannot inline instance in external module: {:?}", module_name);
            },
            None => {
                panic!("Module for instance inlining doesn't exist: {:?}", module_name);
            }
        }
    }
}

/// Inline instances within a module
fn inline_module(
    mut module: Module, 
    config: &InlineConfig,
    module_map: &HashMap<Identifier, CircuitModule>
) -> Module {
    // Create namespace for this module to avoid naming conflicts
    let mut namespace = create_namespace(&module);
    
    // Replace statements with inlined versions
    let inlined_stmts = inline_stmts(
        module.stmts,
        &module.name,
        config,
        module_map,
        &mut namespace
    );
    
    module.stmts = inlined_stmts;
    module
}

/// Create a namespace from a module to track used names
fn create_namespace(module: &Module) -> HashSet<String> {
    let mut names = HashSet::new();
    
    // Add port names
    for port in &module.ports {
        if let Port::Input(id, _, _) | Port::Output(id, _, _) = &**port {
            if let Identifier::Name(name) = id {
                names.insert(name.clone());
            }
        }
    }
    
    // Add names from statements
    collect_names_from_stmts(&module.stmts, &mut names);
    
    names
}

/// Collect names declared in statements recursively
fn collect_names_from_stmts(stmts: &[Box<Stmt>], names: &mut HashSet<String>) {
    for stmt in stmts {
        match &**stmt {
            Stmt::Wire(id, _, _) | 
            Stmt::Reg(id, _, _, _) |
            Stmt::RegReset(id, _, _, _, _, _) |
            Stmt::Node(id, _, _) => {
                if let Identifier::Name(name) = id {
                    names.insert(name.clone());
                }
            },
            Stmt::ChirrtlMemory(mem) => {
                match mem {
                    ChirrtlMemory::SMem(id, _, _, _) |
                    ChirrtlMemory::CMem(id, _, _) => {
                        if let Identifier::Name(name) = id {
                            names.insert(name.clone());
                        }
                    }
                }
            },
            Stmt::ChirrtlMemoryPort(port) => {
                match port {
                    ChirrtlMemoryPort::Read(id, _, _, _, _) |
                    ChirrtlMemoryPort::Write(id, _, _, _, _) |
                    ChirrtlMemoryPort::Infer(id, _, _, _, _) => {
                        if let Identifier::Name(name) = id {
                            names.insert(name.clone());
                        }
                    }
                }
            },
            Stmt::Inst(id, _, _) => {
                if let Identifier::Name(name) = id {
                    names.insert(name.clone());
                }
            },
            Stmt::When(_, _, then_stmts, else_stmts) => {
                collect_names_from_stmts(then_stmts, names);
                if let Some(else_stmts) = else_stmts {
                    collect_names_from_stmts(else_stmts, names);
                }
            },
            _ => {}
        }
    }
}

/// Inline statements, replacing instances with their module bodies
fn inline_stmts(
    stmts: Vec<Box<Stmt>>,
    current_module: &Identifier,
    config: &InlineConfig,
    module_map: &HashMap<Identifier, CircuitModule>,
    namespace: &mut HashSet<String>
) -> Vec<Box<Stmt>> {
    let mut result = Vec::new();
    
    for stmt in stmts {
        match &*stmt {
            Stmt::Inst(inst_name, mod_name, info) => {
                // Check if this instance should be inlined
                let should_inline = config.modules.contains(mod_name) ||
                    config.instances.get(current_module)
                        .map(|insts| insts.contains(inst_name))
                        .unwrap_or(false);
                
                if should_inline {
                    // Get the module to inline
                    if let Some(CircuitModule::Module(to_inline)) = module_map.get(mod_name) {
                        // Generate a unique prefix for inlined names
                        let prefix = generate_prefix(inst_name, namespace);
                        
                        // Create wire declarations for each port
                        let port_wires = create_port_wires(to_inline, &prefix, info, namespace);
                        result.extend(port_wires);
                        
                        // Inline the module body with renamed references
                        let inlined_body = inline_module_body(
                            to_inline, 
                            inst_name,
                            &prefix,
                            current_module, 
                            config,
                            module_map,
                            namespace,
                            info
                        );
                        
                        result.extend(inlined_body);
                    }
                } else {
                    // Keep the instance unchanged
                    result.push(stmt);
                }
            },
            Stmt::When(cond, info, then_stmts, else_stmts) => {
                // Recursively inline in conditional branches
                let inlined_then = inline_stmts(
                    then_stmts.clone(), 
                    current_module,
                    config,
                    module_map,
                    namespace
                );
                
                let inlined_else = else_stmts.as_ref().map(|stmts| {
                    inline_stmts(
                        stmts.clone(),
                        current_module,
                        config,
                        module_map,
                        namespace
                    )
                });
                
                let new_when = Stmt::When(
                    cond.clone(), 
                    info.clone(), 
                    inlined_then,
                    inlined_else
                );
                
                result.push(Box::new(new_when));
            },
            _ => {
                // Keep other statements unchanged
                result.push(stmt);
            }
        }
    }
    
    result
}

/// Generate a unique prefix for inlined module
fn generate_prefix(
    inst_name: &Identifier, 
    namespace: &mut HashSet<String>
) -> String {
    let base_prefix = match inst_name {
        Identifier::Name(name) => format!("{}_", name),
        Identifier::ID(id) => format!("_{}_", id),
    };
    
    let mut prefix = base_prefix.clone();
    let mut counter = 0;
    
    // Ensure the prefix is unique
    while namespace.iter().any(|name| name.starts_with(&prefix)) {
        counter += 1;
        prefix = format!("{}{}__", base_prefix, counter);
    }
    
    prefix
}

/// Create wire declarations for module ports
fn create_port_wires(
    module: &Module,
    prefix: &str,
    info: &Info,
    namespace: &mut HashSet<String>
) -> Vec<Box<Stmt>> {
    let mut wires = Vec::new();
    
    for port in &module.ports {
        match &**port {
            Port::Input(id, tpe, _) | Port::Output(id, tpe, _) => {
                let prefixed_name = match id {
                    Identifier::Name(name) => {
                        let prefixed = format!("{}{}", prefix, name);
                        namespace.insert(prefixed.clone());
                        Identifier::Name(prefixed)
                    },
                    Identifier::ID(x) => {
                        Identifier::ID(x.clone())
                    }
                };
                
                let wire = Stmt::Wire(
                    prefixed_name,
                    tpe.clone(),
                    info.clone()
                );
                
                wires.push(Box::new(wire));
            }
        }
    }
    
    wires
}

/// Inline a module body, renaming references appropriately
fn inline_module_body(
    module: &Module,
    inst_name: &Identifier,
    prefix: &str,
    current_module: &Identifier,
    config: &InlineConfig,
    module_map: &HashMap<Identifier, CircuitModule>,
    namespace: &mut HashSet<String>,
    info: &Info
) -> Vec<Box<Stmt>> {
    // Create a map from original names to prefixed names
    let mut name_map = HashMap::new();
    
    // First, add port mappings
    for port in &module.ports {
        match &**port {
            Port::Input(id, _, _) | Port::Output(id, _, _) => {
                if let Identifier::Name(name) = id {
                    let prefixed = format!("{}{}", prefix, name);
                    name_map.insert(name.clone(), prefixed);
                }
            }
        }
    }
    
    // Now transform and rename all statements
    let mut inlined_stmts = Vec::new();
    
    for stmt in &module.stmts {
        let renamed_stmt = rename_stmt(
            &stmt,
            prefix,
            &mut name_map,
            namespace,
            inst_name
        );
        
        // Further inline any instances within this inlined module
        match &*renamed_stmt {
            Stmt::Inst(_, _, _) => {
                let nested_inlined = inline_stmts(
                    vec![renamed_stmt],
                    current_module,
                    config,
                    module_map,
                    namespace
                );
                inlined_stmts.extend(nested_inlined);
            },
            Stmt::When(cond, when_info, then_stmts, else_stmts) => {
                let inlined_then = inline_stmts(
                    then_stmts.clone(),
                    current_module,
                    config,
                    module_map,
                    namespace
                );
                
                let inlined_else = else_stmts.as_ref().map(|stmts| {
                    inline_stmts(
                        stmts.clone(),
                        current_module,
                        config,
                        module_map,
                        namespace
                    )
                });
                
                let new_when = Box::new(Stmt::When(
                    cond.clone(),
                    when_info.clone(),
                    inlined_then,
                    inlined_else
                ));
                
                inlined_stmts.push(new_when);
            },
            _ => {
                inlined_stmts.push(renamed_stmt);
            }
        }
    }
    
    // Connect instance ports to created wires
    let port_connects = create_port_connections(
        module,
        inst_name,
        prefix,
        &name_map,
        info
    );
    
    inlined_stmts.extend(port_connects);
    
    inlined_stmts
}

/// Rename a statement and all its contained references
fn rename_stmt(
    stmt: &Stmt,
    prefix: &str,
    name_map: &mut HashMap<String, String>,
    namespace: &mut HashSet<String>,
    inst_name: &Identifier
) -> Box<Stmt> {
    match stmt {
        Stmt::Wire(id, tpe, info) => {
            let (new_id, _) = rename_identifier(id, prefix, name_map, namespace);
            Box::new(Stmt::Wire(new_id, tpe.clone(), info.clone()))
        },
        Stmt::Reg(id, tpe, clk, info) => {
            let (new_id, _) = rename_identifier(id, prefix, name_map, namespace);
            let new_clk = rename_expr(clk, name_map);
            Box::new(Stmt::Reg(new_id, tpe.clone(), new_clk, info.clone()))
        },
        Stmt::RegReset(id, tpe, clk, rst, init, info) => {
            let (new_id, _) = rename_identifier(id, prefix, name_map, namespace);
            let new_clk = rename_expr(clk, name_map);
            let new_rst = rename_expr(rst, name_map);
            let new_init = rename_expr(init, name_map);
            Box::new(Stmt::RegReset(
                new_id, tpe.clone(), new_clk, new_rst, new_init, info.clone()
            ))
        },
        Stmt::ChirrtlMemory(mem) => {
            match mem {
                ChirrtlMemory::SMem(id, tpe, ruw, info) => {
                    let (new_id, _) = rename_identifier(id, prefix, name_map, namespace);
                    Box::new(Stmt::ChirrtlMemory(
                        ChirrtlMemory::SMem(new_id, tpe.clone(), ruw.clone(), info.clone())
                    ))
                },
                ChirrtlMemory::CMem(id, tpe, info) => {
                    let (new_id, _) = rename_identifier(id, prefix, name_map, namespace);
                    Box::new(Stmt::ChirrtlMemory(
                        ChirrtlMemory::CMem(new_id, tpe.clone(), info.clone())
                    ))
                }
            }
        },
        Stmt::ChirrtlMemoryPort(port) => {
            match port {
                ChirrtlMemoryPort::Read(port_id, mem_id, addr, clk, info) => {
                    let (new_port_id, _) = rename_identifier(port_id, prefix, name_map, namespace);
                    let new_mem_id = rename_memory_id(mem_id, name_map);
                    let new_addr = rename_expr(addr, name_map);
                    let new_clk = rename_reference(clk, name_map);
                    
                    Box::new(Stmt::ChirrtlMemoryPort(
                        ChirrtlMemoryPort::Read(
                            new_port_id, new_mem_id, new_addr, new_clk, info.clone()
                        )
                    ))
                },
                ChirrtlMemoryPort::Write(port_id, mem_id, addr, clk, info) => {
                    let (new_port_id, _) = rename_identifier(port_id, prefix, name_map, namespace);
                    let new_mem_id = rename_memory_id(mem_id, name_map);
                    let new_addr = rename_expr(addr, name_map);
                    let new_clk = rename_reference(clk, name_map);
                    
                    Box::new(Stmt::ChirrtlMemoryPort(
                        ChirrtlMemoryPort::Write(
                            new_port_id, new_mem_id, new_addr, new_clk, info.clone()
                        )
                    ))
                },
                ChirrtlMemoryPort::Infer(port_id, mem_id, addr, clk, info) => {
                    let (new_port_id, _) = rename_identifier(port_id, prefix, name_map, namespace);
                    let new_mem_id = rename_memory_id(mem_id, name_map);
                    let new_addr = rename_expr(addr, name_map);
                    let new_clk = rename_reference(clk, name_map);
                    
                    Box::new(Stmt::ChirrtlMemoryPort(
                        ChirrtlMemoryPort::Infer(
                            new_port_id, new_mem_id, new_addr, new_clk, info.clone()
                        )
                    ))
                }
            }
        },
        Stmt::Inst(id, module_id, info) => {
            let (new_id, _) = rename_identifier(id, prefix, name_map, namespace);
            Box::new(Stmt::Inst(new_id, module_id.clone(), info.clone()))
        },
        Stmt::Node(id, expr, info) => {
            let (new_id, _) = rename_identifier(id, prefix, name_map, namespace);
            let new_expr = rename_expr(expr, name_map);
            Box::new(Stmt::Node(new_id, new_expr, info.clone()))
        },
        Stmt::Connect(lhs, rhs, info) => {
            let new_lhs = rename_expr(lhs, name_map);
            let new_rhs = rename_expr(rhs, name_map);
            Box::new(Stmt::Connect(new_lhs, new_rhs, info.clone()))
        },
        Stmt::Invalidate(expr, info) => {
            let new_expr = rename_expr(expr, name_map);
            Box::new(Stmt::Invalidate(new_expr, info.clone()))
        },
        Stmt::When(cond, info, then_stmts, else_stmts) => {
            let new_cond = rename_expr(cond, name_map);
            
            let new_then = then_stmts.iter()
                .map(|s| rename_stmt(s, prefix, name_map, namespace, inst_name))
                .collect();
            
            let new_else = else_stmts.as_ref().map(|stmts| {
                stmts.iter()
                    .map(|s| rename_stmt(s, prefix, name_map, namespace, inst_name))
                    .collect()
            });
            
            Box::new(Stmt::When(
                new_cond, info.clone(), new_then, new_else
            ))
        },
        Stmt::Printf(name_opt, clk, en, msg, args_opt, info) => {
            let new_name_opt = name_opt.as_ref().map(|id| {
                let (new_id, _) = rename_identifier(id, prefix, name_map, namespace);
                new_id
            });
            
            let new_clk = rename_expr(clk, name_map);
            let new_en = rename_expr(en, name_map);
            
            let new_args_opt = args_opt.as_ref().map(|args| {
                args.iter()
                    .map(|arg| Box::new(rename_expr(arg, name_map)))
                    .collect()
            });
            
            Box::new(Stmt::Printf(
                new_name_opt, new_clk, new_en, msg.clone(), new_args_opt, info.clone()
            ))
        },
        Stmt::Assert(name_opt, clk, cond, en, msg, info) => {
            let new_name_opt = name_opt.as_ref().map(|id| {
                let (new_id, _) = rename_identifier(id, prefix, name_map, namespace);
                new_id
            });
            
            let new_clk = rename_expr(clk, name_map);
            let new_cond = rename_expr(cond, name_map);
            let new_en = rename_expr(en, name_map);
            
            Box::new(Stmt::Assert(
                new_name_opt, new_clk, new_cond, new_en, msg.clone(), info.clone()
            ))
        },
        Stmt::Skip(info) => Box::new(Stmt::Skip(info.clone())),
    }
}

/// Rename an identifier and update the name map
fn rename_identifier(
    id: &Identifier,
    prefix: &str,
    name_map: &mut HashMap<String, String>,
    namespace: &mut HashSet<String>
) -> (Identifier, bool) {
    match id {
        Identifier::Name(name) => {
            let prefixed = format!("{}{}", prefix, name);
            name_map.insert(name.clone(), prefixed.clone());
            namespace.insert(prefixed.clone());
            (Identifier::Name(prefixed), true)
        },
        Identifier::ID(x) => (Identifier::ID(x.clone()), false),
    }
}

/// Rename a memory identifier
fn rename_memory_id(
    id: &Identifier,
    name_map: &HashMap<String, String>
) -> Identifier {
    match id {
        Identifier::Name(name) => {
            if let Some(new_name) = name_map.get(name) {
                Identifier::Name(new_name.clone())
            } else {
                id.clone()
            }
        },
        _ => id.clone(),
    }
}

/// Rename a reference
fn rename_reference(
    reference: &Reference,
    name_map: &HashMap<String, String>
) -> Reference {
    match reference {
        Reference::Ref(id) => {
            match id {
                Identifier::Name(name) => {
                    if let Some(new_name) = name_map.get(name) {
                        Reference::Ref(Identifier::Name(new_name.clone()))
                    } else {
                        reference.clone()
                    }
                },
                _ => reference.clone(),
            }
        },
        Reference::RefDot(parent, field) => {
            let new_parent = Box::new(rename_reference(parent, name_map));
            Reference::RefDot(new_parent, field.clone())
        },
        Reference::RefIdxInt(parent, idx) => {
            let new_parent = Box::new(rename_reference(parent, name_map));
            Reference::RefIdxInt(new_parent, idx.clone())
        },
        Reference::RefIdxExpr(parent, expr) => {
            let new_parent = Box::new(rename_reference(parent, name_map));
            let new_expr = Box::new(rename_expr(expr, name_map));
            Reference::RefIdxExpr(new_parent, new_expr)
        },
    }
}

/// Rename an expression
fn rename_expr(
    expr: &Expr,
    name_map: &HashMap<String, String>
) -> Expr {
    match expr {
        Expr::Reference(reference) => {
            Expr::Reference(rename_reference(reference, name_map))
        },
        Expr::Mux(cond, te, fe) => {
            let new_cond = Box::new(rename_expr(cond, name_map));
            let new_te = Box::new(rename_expr(te, name_map));
            let new_fe = Box::new(rename_expr(fe, name_map));
            Expr::Mux(new_cond, new_te, new_fe)
        },
        Expr::ValidIf(cond, te) => {
            let new_cond = Box::new(rename_expr(cond, name_map));
            let new_te = Box::new(rename_expr(te, name_map));
            Expr::ValidIf(new_cond, new_te)
        },
        Expr::PrimOp2Expr(op, a, b) => {
            let new_a = Box::new(rename_expr(a, name_map));
            let new_b = Box::new(rename_expr(b, name_map));
            Expr::PrimOp2Expr(*op, new_a, new_b)
        },
        Expr::PrimOp1Expr(op, a) => {
            let new_a = Box::new(rename_expr(a, name_map));
            Expr::PrimOp1Expr(*op, new_a)
        },
        Expr::PrimOp1Expr1Int(op, a, i) => {
            let new_a = Box::new(rename_expr(a, name_map));
            Expr::PrimOp1Expr1Int(*op, new_a, i.clone())
        },
        Expr::PrimOp1Expr2Int(op, a, i1, i2) => {
            let new_a = Box::new(rename_expr(a, name_map));
            Expr::PrimOp1Expr2Int(*op, new_a, i1.clone(), i2.clone())
        },
        _ => expr.clone(),
    }
}

/// Create connections between instance ports and the wires
fn create_port_connections(
    module: &Module,
    inst_name: &Identifier,
    prefix: &str,
    name_map: &HashMap<String, String>,
    info: &Info
) -> Vec<Box<Stmt>> {
    let mut connects = Vec::new();
    
    for port in &module.ports {
        match &**port {
            Port::Input(id, _, _) => {
                // For input ports, connect instance.port to the created wire
                if let Identifier::Name(name) = id {
                    if let Some(prefixed_name) = name_map.get(name) {
                        // Create lhs: prefixed port wire
                        let lhs = Expr::Reference(Reference::Ref(
                            Identifier::Name(prefixed_name.clone())
                        ));
                        
                        // Create rhs: instance.port reference
                        let rhs = match inst_name {
                            Identifier::Name(inst) => {
                                Expr::Reference(Reference::RefDot(
                                    Box::new(Reference::Ref(
                                        Identifier::Name(inst.clone())
                                    )),
                                    Identifier::Name(name.clone())
                                ))
                            },
                            Identifier::ID(id) => {
                                Expr::Reference(Reference::RefDot(
                                    Box::new(Reference::Ref(
                                        Identifier::ID(id.clone())
                                    )),
                                    Identifier::Name(name.clone())
                                ))
                            }
                        };
                        
                        connects.push(Box::new(Stmt::Connect(lhs, rhs, info.clone())));
                    }
                }
            },
            Port::Output(id, _, _) => {
                // For output ports, connect wire to instance.port
                if let Identifier::Name(name) = id {
                    if let Some(prefixed_name) = name_map.get(name) {
                        // Create lhs: instance.port reference
                        let lhs = match inst_name {
                            Identifier::Name(inst) => {
                                Expr::Reference(Reference::RefDot(
                                    Box::new(Reference::Ref(
                                        Identifier::Name(inst.clone())
                                    )),
                                    Identifier::Name(name.clone())
                                ))
                            },
                            Identifier::ID(id) => {
                                Expr::Reference(Reference::RefDot(
                                    Box::new(Reference::Ref(
                                        Identifier::ID(id.clone())
                                    )),
                                    Identifier::Name(name.clone())
                                ))
                            }
                        };
                        
                        // Create rhs: prefixed port wire
                        let rhs = Expr::Reference(Reference::Ref(
                            Identifier::Name(prefixed_name.clone())
                        ));
                        
                        connects.push(Box::new(Stmt::Connect(lhs, rhs, info.clone())));
                    }
                }
            }
        }
    }
    
    connects
}
