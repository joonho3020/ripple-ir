use logos::Logos;

#[derive(Logos, Debug, PartialEq)]
pub enum Token {
    #[token("module")]
    Module,

    #[token("input")]
    Input,

    #[token("output")]
    Output,

    #[token("wire")]
    Wire,

    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", priority = 2)]
    Identifier,

    #[regex(r"[0-9]+", priority = 1)]
    Number,

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token("{")]
    LBrace,

    #[token("}")]
    RBrace,

    #[token(";")]
    Semicolon,

    #[regex(r"[ \t\n\f]+", logos::skip)]
    Whitespace,

    #[error]
    Error,
}
