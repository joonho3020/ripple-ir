pub mod typetree;
pub mod subtree;
pub mod tnode;
pub mod tedge;
pub mod path;
pub mod typetree_graphviz;

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
                                assert_eq!(v.node_name(&Identifier::Name("io".to_string()), TypeTreeNodeIndex::from(1u32)).to_string(), "io.value1".to_string());
                                assert_eq!(v.node_name(&Identifier::Name("io".to_string()), TypeTreeNodeIndex::from(2u32)).to_string(), "io.value2".to_string());
                                assert_eq!(v.node_name(&Identifier::Name("io".to_string()), TypeTreeNodeIndex::from(3u32)).to_string(), "io.loadingValues".to_string());
                                assert_eq!(v.node_name(&Identifier::Name("io".to_string()), TypeTreeNodeIndex::from(4u32)).to_string(), "io.outputGCD".to_string());
                                assert_eq!(v.node_name(&Identifier::Name("io".to_string()), TypeTreeNodeIndex::from(5u32)).to_string(), "io.outputValid".to_string());
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

    fn nested_bundle_output_port_type() -> Result<Type, RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/NestedBundle.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        match port.as_ref() {
                            Port::Output(_name, tpe, _info) => {
                                return Ok(tpe.clone());
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
        Err(RippleIRErr::MiscError("Output port not found in NestedBundle".to_string()))
    }

    #[test]
    fn check_subtree_root() -> Result<(), RippleIRErr> {
        let tpe = nested_bundle_output_port_type()?;
        let typetree = TypeTree::build_from_type(&tpe, TypeDirection::Outgoing);

        let _ = typetree.export_graphviz("./test-outputs/NestedBundle.typetree.pdf", None, None, false);
        let v = typetree.view().unwrap();

        let root = Reference::Ref(Identifier::Name("io".to_string()));
        let subtree_root = v.subtree_root(&root);
        assert_eq!(subtree_root, Some(TypeTreeNodeIndex::from(0u32)));

        let g = Reference::RefDot(Box::new(root), Identifier::Name("g".to_string()));
        let subtree_root = v.subtree_root(&g);
        assert_eq!(subtree_root, Some(TypeTreeNodeIndex::from(1u32)));

        let g_name = v.node_name(&Identifier::Name("io".to_string()), subtree_root.unwrap());
        assert_eq!(g_name.to_string(), "io.g".to_string());

        let g1 = Reference::RefIdxInt(Box::new(g), Int::from(1));
        let subtree_root = v.subtree_root(&g1);
        assert_eq!(subtree_root, Some(TypeTreeNodeIndex::from(24u32)));

        let g1_name = v.node_name(&Identifier::Name("io".to_string()), subtree_root.unwrap());
        assert_eq!(g1_name.to_string(), "io.g[1]".to_string());

        let g1f = Reference::RefDot(Box::new(g1), Identifier::Name("f".to_string()));
        let subtree_root = v.subtree_root(&g1f);
        assert_eq!(subtree_root, Some(TypeTreeNodeIndex::from(42u32)));

        let g1f_name = v.node_name(&Identifier::Name("io".to_string()), subtree_root.unwrap());
        assert_eq!(g1f_name.to_string(), "io.g[1].f".to_string());

        let subtree_leaves = v.subtree_leaves(&g1f);
        assert_eq!(subtree_leaves, vec![TypeTreeNodeIndex::from(45u32), TypeTreeNodeIndex::from(44u32), TypeTreeNodeIndex::from(43u32)]);
        Ok(())
    }

    #[test]
    fn check_subtree_leaves_with_path() -> Result<(), RippleIRErr> {
        let tpe = nested_bundle_output_port_type()?;
        let typetree = TypeTree::build_from_type(&tpe, TypeDirection::Outgoing);

        let root = Reference::Ref(Identifier::Name("io".to_string()));
        let g = Reference::RefDot(Box::new(root), Identifier::Name("g".to_string()));
        let g1 = Reference::RefIdxInt(Box::new(g), Int::from(1));
        let g1f = Reference::RefDot(Box::new(g1.clone()), Identifier::Name("f".to_string()));

        let v = typetree.view().unwrap();
        let leaves_with_path = v.subtree_leaves_with_path(&g1f);

        let mut expect: IndexMap<TypeTreeNodePath, TypeTreeNodeIndex> = IndexMap::new();
        expect.insert(
            TypeTreeNodePath::new(
                TypeDirection::Outgoing,
                TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(2)))),
                Some(Reference::Ref(Identifier::ID(Int::from(2))))),
                TypeTreeNodeIndex::from(45u32));

        expect.insert(
            TypeTreeNodePath::new(
                TypeDirection::Outgoing,
                TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(2)))),
                Some(Reference::Ref(Identifier::ID(Int::from(1))))),
                TypeTreeNodeIndex::from(44u32));

        expect.insert(
            TypeTreeNodePath::new(
                TypeDirection::Outgoing,
                TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(2)))),
                Some(Reference::Ref(Identifier::ID(Int::from(0))))),
                TypeTreeNodeIndex::from(43u32));
        assert_eq!(leaves_with_path, expect);
        Ok(())
    }

    #[test]
    fn to_type() -> Result<(), RippleIRErr> {
        let tpe = nested_bundle_output_port_type()?;
        let typetree = TypeTree::build_from_type(&tpe, TypeDirection::Outgoing);
        let reconstructed_tpe = typetree.to_type();
        assert_eq!(tpe, reconstructed_tpe);
        Ok(())
    }
}
