pub mod typetree;
pub mod subtree;
pub mod tnode;
pub mod tedge;
pub mod path;

#[cfg(test)]
mod test {
    use chirrtl_parser::ast::*;
    use chirrtl_parser::parse_circuit;
    use indexmap::IndexMap;
    use crate::common::RippleIRErr;
    use crate::common::graphviz::*;
    use crate::ir::typetree::typetree::*;
    use crate::ir::typetree::tnode::*;
    use crate::ir::typetree::path::*;

    #[test]
    fn check_gcd_name() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/GCD.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        match port.as_ref() {
                            Port::Output(_name, tpe, _info) => {
                                let tt = TypeTree::build_from_type(tpe, TypeDirection::Outgoing);
                                let v = tt.view().unwrap();
                                assert_eq!(v.node_name(&Identifier::Name("io".to_string()), TTreeNodeIndex::from(1)).to_string(), "io.value1".to_string());
                                assert_eq!(v.node_name(&Identifier::Name("io".to_string()), TTreeNodeIndex::from(2)).to_string(), "io.value2".to_string());
                                assert_eq!(v.node_name(&Identifier::Name("io".to_string()), TTreeNodeIndex::from(3)).to_string(), "io.loadingValues".to_string());
                                assert_eq!(v.node_name(&Identifier::Name("io".to_string()), TTreeNodeIndex::from(4)).to_string(), "io.outputGCD".to_string());
                                assert_eq!(v.node_name(&Identifier::Name("io".to_string()), TTreeNodeIndex::from(5)).to_string(), "io.outputValid".to_string());
                            }
                            _ => {
                            }
                        };
                    }

                }
                CircuitModule::ExtModule(_e) => {
                }
            }
        }
        Ok(())
    }

    #[test]
    fn print_type_tree() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/NestedBundle.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        let port_typetree = match port.as_ref() {
                            Port::Input(_name, tpe, _info) => {
                                TypeTree::build_from_type(tpe, TypeDirection::Incoming)
                            }
                            Port::Output(_name, tpe, _info) => {
                                TypeTree::build_from_type(tpe, TypeDirection::Outgoing)
                            }
                        };
                        let v = port_typetree.view().unwrap();
                        v.print_tree();
                    }

                }
                CircuitModule::ExtModule(_e) => {
                }
            }
        }
        return Ok(());
    }

    #[test]
    fn check_subtree_root() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/NestedBundle.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        match port.as_ref() {
                            Port::Output(_name, tpe, _info) => {
                                let typetree = TypeTree::build_from_type(tpe, TypeDirection::Outgoing);
                                let _ = typetree.export_graphviz("./test-outputs/NestedBundle.typetree.pdf", None, None, false);
                                let v = typetree.view().unwrap();

                                let root = Reference::Ref(Identifier::Name("io".to_string()));
                                let subtree_root = v.subtree_root(&root);
                                assert_eq!(subtree_root, Some(TTreeNodeIndex::from(0)));

                                let g = Reference::RefDot(Box::new(root), Identifier::Name("g".to_string()));
                                let subtree_root = v.subtree_root(&g);
                                assert_eq!(subtree_root, Some(TTreeNodeIndex::from(1)));

                                let g_name = v.node_name(&Identifier::Name("io".to_string()), subtree_root.unwrap());
                                assert_eq!(g_name.to_string(), "io.g".to_string());

                                let g1 = Reference::RefIdxInt(Box::new(g), Int::from(1));
                                let subtree_root = v.subtree_root(&g1);
                                assert_eq!(subtree_root, Some(TTreeNodeIndex::from(24)));

                                let g1_name = v.node_name(&Identifier::Name("io".to_string()), subtree_root.unwrap());
                                assert_eq!(g1_name.to_string(), "io.g[1]".to_string());

                                let g1f = Reference::RefDot(Box::new(g1), Identifier::Name("f".to_string()));
                                let subtree_root = v.subtree_root(&g1f);
                                assert_eq!(subtree_root, Some(TTreeNodeIndex::from(42)));

                                let g1f_name = v.node_name(&Identifier::Name("io".to_string()), subtree_root.unwrap());
                                assert_eq!(g1f_name.to_string(), "io.g[1].f".to_string());

                                let subtree_leaves = v.subtree_leaves(&g1f);
                                assert_eq!(subtree_leaves, vec![TTreeNodeIndex::from(45), TTreeNodeIndex::from(44), TTreeNodeIndex::from(43)]);
                            }
                            _ => {
                            }
                        };
                    }

                }
                CircuitModule::ExtModule(_e) => {
                }
            }
        }
        return Ok(());
    }

    #[test]
    fn check_subtree_leaves_with_path() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/NestedBundle.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        match port.as_ref() {
                            Port::Output(_name, tpe, _info) => {
                                let typetree = TypeTree::build_from_type(tpe, TypeDirection::Outgoing);

                                let _ = typetree.export_graphviz("./test-outputs/NestedBundle.typetree.pdf", None, None, false);

                                let root = Reference::Ref(Identifier::Name("io".to_string()));
                                let g = Reference::RefDot(Box::new(root), Identifier::Name("g".to_string()));
                                let g1 = Reference::RefIdxInt(Box::new(g), Int::from(1));
                                let g1f = Reference::RefDot(Box::new(g1.clone()), Identifier::Name("f".to_string()));

                                let v = typetree.view().unwrap();
                                let leaves_with_path = v.subtree_leaves_with_path(&g1f);

                                let mut expect: IndexMap<TypeTreeNodePath, TTreeNodeIndex> = IndexMap::new();
                                expect.insert(
                                    TypeTreeNodePath::new(
                                        TypeDirection::Outgoing,
                                        TypeTreeNodeType::Ground(GroundType::UInt),
                                        Some(Reference::Ref(Identifier::ID(Int::from(2))))),
                                    TTreeNodeIndex::from(45));

                                expect.insert(
                                    TypeTreeNodePath::new(
                                        TypeDirection::Outgoing,
                                        TypeTreeNodeType::Ground(GroundType::UInt),
                                        Some(Reference::Ref(Identifier::ID(Int::from(1))))),
                                    TTreeNodeIndex::from(44));

                                expect.insert(
                                    TypeTreeNodePath::new(
                                        TypeDirection::Outgoing,
                                        TypeTreeNodeType::Ground(GroundType::UInt),
                                        Some(Reference::Ref(Identifier::ID(Int::from(0))))),
                                    TTreeNodeIndex::from(43));
                                assert_eq!(leaves_with_path, expect);
                            }
                            _ => {
                            }
                        };
                    }

                }
                CircuitModule::ExtModule(_e) => {
                }
            }
        }
        return Ok(());
    }
}
