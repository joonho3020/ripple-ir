use logos::{Logos, Lexer};
use std::collections::VecDeque;


#[derive(Logos, Debug, PartialEq)]
pub enum Token {
    Indent,
    Dedent,
    Info,
    ID,

    #[token(" ")]
    Space,

    #[token("\t")]
    Tab,

    #[token("\n")]
    Newline,

    #[regex("0b[01]+|0o[0-7]+|0d[0-9]+|0h[0-9A-Fa-f]+", priority = 2)]
    RadixInt,

    #[regex("-?[0-9]+", priority = 3)]
    IntegerDec,

    #[regex("[_A-Za-z][_A-Za-z0-9]*", priority = 1)]
    Identifier,

    #[regex(r#""([^"\\]|\\.)*""#)]
    String,

// #[regex(r"[{}\(\)\[\],.:;]")]
// Symbol,

    #[token("/")]
    Slash,

    #[token("[")]
    LeftSquare,

    #[token("]")]
    RightSquare,

    #[token("<")]
    LeftAngle,

    #[token(">")]
    RightAngle,

    #[token("{")]
    LeftBracket,

    #[token("}")]
    RightBracket,

    #[token("(")]
    LeftParenthesis,

    #[token(")")]
    RightParenthesis,

    #[token("@")]
    AtSymbol,

    #[token("<<")]
    DoubleLeft,

    #[token(">>")]
    DoubleRight,


    #[token("Clock")]
    Clock,

    #[token("Reset")]
    Reset,

    #[token("AsyncReset")]
    AsyncReset,

    #[token("UInt")]
    UInt,

    #[token("SInt")]
    SInt,

    #[token("probe")]
    ProbeType,

    #[token("Probe")]
    Probe,

    #[token("Analog")]
    Analog,

    #[token("Fixed")]
    Fixed,

    #[token("flip")]
    Flip,

    #[regex("add|sub|mul|div|rem|lt|leq|gt|geq|eq|neq|dshl|dshr|and|or|xor|cat")]
    E2Op,

    #[regex("asUInt|asSInt|asClock|asAsyncReset|cvt|neg|not|andr|orr|xorr")]
    E1Op,

    #[regex("(pad|shl|shr|head|tail)[(]")]
    E1I1Op,

    #[regex("bits[(]")]
    E1I2Op,

    #[token("mux")]
    Mux,

    #[token("validif")]
    ValidIf,

    #[token("mem")]
    Mem,

    #[token("smem")]
    SMem,

    #[token("cmem")]
    CMem,

    #[token("write")]
    Write,

    #[token("read")]
    Read,

    #[token("infer")]
    Infer,

    #[token("mport")]
    Mport,

    #[token("data-type")]
    DataType,

    #[token("depth")]
    Depth,

    #[token("read-latency")]
    ReadLatency,

    #[token("write-latency")]
    WriteLatency,

    #[token("read-under-write")]
    ReadUnderWrite,

    #[token("reader")]
    Reader,

    #[token("writer")]
    Writer,

    #[token("readwriter")]
    Readwriter,

    #[token("wire")]
    Wire,

    #[token("reg")]
    Reg,

    #[token("regreset")]
    Regreset,

    #[token("inst")]
    Inst,

    #[token("of")]
    Of,

    #[token("node")]
    Node,

    #[token("invalidate")]
    Invalidate,

    #[token("attach")]
    Attach,

    #[token("when")]
    When,

    #[token("else")]
    Else,

    #[token("stop")]
    Stop,

    #[token("printf")]
    Printf,

    #[token("assert")]
    Assert,

    #[token("skip")]
    Skip,

    #[token("input")]
    Input,

    #[token("output")]
    Output,

    #[token("module")]
    Module,

    #[token("extmodule")]
    Extmodule,

    #[token("defname")]
    Defname,

    #[token("parameter")]
    Parameter,

    #[token("intmodule")]
    Intmodule,

    #[token("intrinsic")]
    Intrinsic,

    #[token("FIRRTL")]
    FIRRTL,

    #[token("version")]
    Version,

    #[token("circuit")]
    Circuit,

    #[token("connect")]
    Connect,

    #[token("public")]
    Public,

    #[token("define")]
    Define,

    #[token("const")]
    Const,

    #[regex(r"[.,:=@%<>()\[\]{}]")]
    Symbol,

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
    angle_num: u32,
    square_num: u32,
    bracket_num: u32,
    parenthesis_num: u32,
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
            angle_num: 0,
            square_num: 0,
            bracket_num: 0,
            parenthesis_num: 0,
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
            Token::LeftSquare => {
                self.info_string = String::default();
                None
            }
            Token::RightSquare => {
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
            Token::IntegerDec => {
                if self.angle_num == 0 &&
                    self.square_num == 0 &&
                    self.parenthesis_num == 0 &&
                    self.bracket_num != 0 {
                    Some(TokenString::new(Token::ID, ts.name.unwrap()))
                } else {
                    Some(ts)
                }
            }
            Token::AtSymbol => {
                self.mode = LexerMode::Info;
                None
            }
            Token::LeftAngle => {
                self.angle_num += 1;
                Some(ts)
            }
            Token::RightAngle => {
                self.angle_num -= 1;
                Some(ts)
            }
            Token::LeftBracket => {
                self.bracket_num += 1;
                Some(ts)
            }
            Token::RightBracket => {
                self.bracket_num -= 1;
                Some(ts)
            }
            Token::LeftParenthesis => {
                self.parenthesis_num += 1;
                Some(ts)
            }
            Token::RightParenthesis => {
                self.parenthesis_num -= 1;
                Some(ts)
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
