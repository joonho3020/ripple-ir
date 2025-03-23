use std::fmt::Display;
use graphviz_rust::{
    attributes::{rankdir, EdgeAttributes, GraphAttributes, NodeAttributes},
    cmd::{CommandArg, Format},
    dot_generator::{edge, id, node_id},
    dot_structures::*,
    exec_dot,
    printer::{DotPrinter, PrinterContext}
};
use indexmap::IndexMap;
use petgraph::graph::{EdgeIndex, EdgeIndices, NodeIndex, NodeIndices};
use pdfium_render::prelude::*;
use image::{RgbaImage, DynamicImage};
use gif::{Encoder, Frame, Repeat};
use std::fs::File;
use std::io::BufWriter;
use spinoff::{Spinner, spinners};
use crate::common::RippleIRErr;

/// IndexMap from NodeIndex to a GraphViz node attribute.
/// The attributes are added when passing this value to `export_graphviz`
pub type NodeAttributeMap = IndexMap<NodeIndex, Attribute>;

/// IndexMap from EdgeIndex to a GraphViz node attribute.
/// The attributes are added when passing this value to `export_graphviz`
pub type EdgeAttributeMap = IndexMap<EdgeIndex, Attribute>;

/// By implementing this trait, you can easily export petgraph graphs
/// into a Graphviz dot format
pub trait DefaultGraphVizCore<N, E>
where
    N: Display,
    E: Display,
{
    fn node_indices(self: &Self) -> NodeIndices;
    fn node_weight(self: &Self, id: NodeIndex) -> Option<&N>;
    fn edge_indices(self: &Self) -> EdgeIndices;
    fn edge_endpoints(self: &Self, id: EdgeIndex) -> Option<(NodeIndex, NodeIndex)>;
    fn edge_weight(self: &Self, id: EdgeIndex) -> Option<&E>;

    /// Creates a string that represents contents of a dot file
    fn graphviz_string(
        self: &Self,
        node_attr: Option<&NodeAttributeMap>,
        edge_attr: Option<&EdgeAttributeMap>
    ) -> Result<String, std::io::Error> {
        let mut g = graphviz_rust::dot_structures::Graph::DiGraph {
            id: graphviz_rust::dot_generator::id!(""),
            strict: false,
            stmts: vec![
                Stmt::from(GraphAttributes::rankdir(rankdir::TB)),
                Stmt::from(GraphAttributes::splines(true))
            ]
        };

        for id in self.node_indices() {
            let w = self.node_weight(id).unwrap();

            // Create graphviz node
            let mut gv_node = Node {
                id: NodeId(Id::Plain(id.index().to_string()), None),
                attributes: vec![
                    NodeAttributes::label(format!("\"{}\"", w).to_string())
                ],
            };

            // Add node attribute if it exists
            if let Some(na) = node_attr{
                if na.contains_key(&id) {
                    gv_node.attributes.push(na.get(&id).unwrap().clone());
                }
            }

            g.add_stmt(Stmt::Node(gv_node));
        }

        for id in self.edge_indices() {
            let ep = self.edge_endpoints(id).unwrap();
            let w = self.edge_weight(id).unwrap();

            // Create graphviz edge
            let mut e = edge!(
                node_id!(ep.0.index().to_string()) =>
                node_id!(ep.1.index().to_string()));

            e.attributes.push(
                EdgeAttributes::label(format!("\"{}\"", w).to_string()));

            // Add edge attribute if it exists
            if let Some(ea) = edge_attr {
                if ea.contains_key(&id) {
                    e.attributes.push(ea.get(&id).unwrap().clone());
                }
            }

            g.add_stmt(Stmt::Edge(e));
        }

        // Export to pdf
        let dot = g.print(&mut PrinterContext::new(true, 4, "\n".to_string(), 90));
        Ok(dot)
    }
}

pub trait GraphViz {

    /// Creates a string that represents contents of a dot file
    fn graphviz_string(
        self: &Self,
        node_attr: Option<&NodeAttributeMap>,
        edge_attr: Option<&EdgeAttributeMap>
    ) -> Result<String, std::io::Error>;

    /// Exports the graph into a graphviz pdf output
    /// - path: should include the filename
    /// - node_attr: adds additional node attributes when provided
    /// - edge_attr: adds additional edge attributes when provided
    fn export_graphviz(
        self: &Self,
        path: &str,
        node_attr: Option<&NodeAttributeMap>,
        edge_attr: Option<&EdgeAttributeMap>,
        debug: bool
    ) -> Result<String, std::io::Error> {
        let dot = self.graphviz_string(node_attr, edge_attr)?;
        if debug {
            println!("{}", dot);
        }

        let mut spinner = Spinner::new(spinners::Dots, "Exporting dot file...", None);
        exec_dot(dot.clone(), vec![Format::Pdf.into(), CommandArg::Output(path.to_string())])?;
        spinner.success("Finished exporting dotfile");

        return Ok(dot);
    }

    /// Given a list of pdf files, create a gif from it. Frames are ordered
    /// in the order of `pdf_files`.
    /// - path: path to the output file. should include the filename
    /// - pdf_files: each entry represents a path to a pdf
    fn create_gif(
        self: &Self,
        path: &str,
        pdf_files: &Vec<String>
    ) -> Result<(), RippleIRErr> {
        // Initialize Pdfium
        let pdfium = Pdfium::new(Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./pdfium/lib")).unwrap());

        // Vector to hold rendered images
        let mut images = Vec::new();

         let render_config = PdfRenderConfig::new()
            .set_target_width(8000)
            .set_maximum_height(12000);

        for pdf_path in pdf_files {
            let document = pdfium.load_pdf_from_file(pdf_path, None)?;

            for page_index in 0..document.pages().len() {
                let page = document.pages().get(page_index).unwrap();
                let bitmap = page.render_with_config(&render_config)?;

                // Convert the bitmap to an RgbaImage
                let rgba_image = RgbaImage::from_raw(
                    bitmap.width() as u32,
                    bitmap.height() as u32,
                    bitmap.as_rgba_bytes(),
                ).unwrap();

                images.push(DynamicImage::ImageRgba8(rgba_image));
            }
        }

        let mut spinner = Spinner::new(spinners::Dots, "Creating gif from images...", None);

        // Create a GIF file
        let gif_file = File::create(path).unwrap();
        let mut encoder = Encoder::new(BufWriter::new(gif_file), images[0].width() as u16, images[0].height() as u16, &[]).unwrap();
        encoder.set_repeat(Repeat::Infinite).unwrap();

        // Convert images to GIF frames
        for img in images {
            let img = img.to_rgba8();
            let frame = Frame::from_rgba_speed(img.width() as u16, img.height() as u16, &mut img.clone().into_raw(), 4);
            encoder.write_frame(&frame).unwrap();
        }

        spinner.success("Finished generating gif");

        Ok(())
    }
}
