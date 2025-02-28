use logos::{Logos, Lexer};
use std::collections::VecDeque;


#[derive(Logos, Debug, PartialEq)]
pub enum Token {
    Indent,
    Dedent,
    Info,

    #[token(" ")]
    Space,

    #[token("\t")]
    Tab,

    #[token("\n")]
    Newline,

    #[regex("0b[01]+|0o[0-7]+|0d[0-9]+|0h[0-9A-Fa-f]+", priority = 2)]
    RadixInt,

    #[regex("-?[0-9]+", priority = 3)]
    Integer,

    #[regex("[_A-Za-z][_A-Za-z0-9]*", priority = 1)]
    Identifier,

    #[regex(r#""([^"\\]|\\.)*""#)]
    String,

    #[regex(r"[{}\(\)\[\],.:;]")]
    Symbol,

    #[token("/")]
    Slash,

    #[token("[")]
    LeftSquareBracket,

    #[token("]")]
    RightSquareBracket,

    #[token("@")]
    AtSymbol,

    #[token("FIRRTL")]
    FIRRTL,

    #[token("version")]
    Version,

    #[token("module")]
    Module,

    #[token("circuit")]
    Circuit,

    #[error]
    Error,
}


#[derive(Default, Debug, Clone)]
enum LexerMode {
    #[default]
    Indent,
    IntId,
    Info,
    Anno,
    Normal,
}

#[derive(Debug)]
pub struct TokenString {
    pub token: Token,
    pub name: Option<String>,
}

impl From<Token> for TokenString {
    fn from(token: Token) -> Self {
        Self {
            token,
            name: None
        }
    }
}

impl TokenString {
    fn new(token: Token, name: String) -> Self {
        Self {
            token,
            name: Some(name)
        }
    }
}

#[derive(Debug)]
pub struct FIRRTLLexer<'a> {
    lexer: Lexer<'a, Token>,
    tokens: VecDeque<TokenString>,
    mode: LexerMode,
    indent_levels: Vec<u32>,
    cur_indent: u32,
    info_string: String,
}

impl<'a> FIRRTLLexer<'a> {
    const TAB_WIDTH: u32 = 2;

    pub fn new(input: &'a str) -> Self {
        Self {
            lexer: Token::lexer(input),
            tokens: VecDeque::new(),
            indent_levels: vec![0],
            mode: LexerMode::Indent,
            cur_indent: 0,
            info_string: String::default(),
        }
    }

    fn indent_mode(&mut self) -> Option<TokenString> {
        let ts = self.tokens.pop_front().unwrap();
        match ts.token {
            Token::Space => {
                self.cur_indent += 1;
                None
            }
            Token::Tab => {
                self.cur_indent = (self.cur_indent + Self::TAB_WIDTH) & !(Self::TAB_WIDTH - 1);
                None
            }
            Token::Newline => {
                self.cur_indent = 0;
                None
            }
            _ => {
                self.tokens.push_front(ts);

// println!("INDENT MODE cur_indent: {} top: {:?}",
// self.cur_indent,
// self.indent_levels.last());

                let lvl = *self.indent_levels.last().unwrap();
                if self.cur_indent > lvl {
                    self.mode = LexerMode::Normal;
                    self.indent_levels.push(self.cur_indent);
                    return Some(TokenString::from(Token::Indent));
                } else if self.cur_indent < lvl {
                    self.indent_levels.pop();
                    return Some(TokenString::from(Token::Dedent));
                } else {
                    self.mode = LexerMode::Normal;
                    None
                }
            }
        }
    }

    fn info_mode(&mut self) -> Option<TokenString> {
        let ts = self.tokens.pop_front().unwrap();
        match ts.token {
            Token::LeftSquareBracket => {
                self.info_string = String::default();
                None
            }
            Token::RightSquareBracket => {
                self.mode = LexerMode::Normal;
                Some(TokenString::new(Token::Info, self.info_string.clone()))
            }
            _ => {
                self.info_string.push_str(&ts.name.unwrap());
                None
            }
        }
    }

    fn normal_mode(&mut self) -> Option<TokenString> {
        let ts = self.tokens.pop_front().unwrap();
        match ts.token {
            Token::Newline => {
                self.cur_indent = 0;
                self.mode = LexerMode::Indent;
                None
            }
            Token::Space => {
                None
            }
            Token::AtSymbol => {
                self.mode = LexerMode::Info;
                None
            }
            _ => {
                Some(ts)
            }
        }

    }

    fn try_push(&mut self) {
        match self.lexer.next() {
            Some(token) => {
                self.tokens.push_back(TokenString::new(token, self.lexer.slice().to_string()));
            }
            _ => { }
        }
    }

    pub fn next(&mut self) -> Option<TokenString> {
        self.try_push();

        while !self.tokens.is_empty() {
// println!("mode: {:?}, token: {:?}", self.mode, self.tokens.front());

            match self.mode {
                LexerMode::Indent => {
                    match self.indent_mode() {
                        Some(ts) => {
                            return Some(ts)
                        }
                        _ => {
                            self.try_push();
                            continue;
                        }
                    }
                }
                LexerMode::IntId => {
                }
                LexerMode::Info => {
                    match self.info_mode() {
                        Some(ts) => {
                            return Some(ts)
                        }
                        _ => {
                            self.try_push();
                            continue;
                        }
                    }
                }
                LexerMode::Anno => {
                }
                LexerMode::Normal => {
                    match self.normal_mode() {
                        Some(ts) => {
                            return Some(ts)
                        }
                        _ => {
                            self.try_push();
                            continue;
                        }
                    }
                }
            }
        }
        None
    }
}
