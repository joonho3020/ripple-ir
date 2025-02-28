mod lexer;


#[cfg(test)]
mod test {
    use crate::lexer::*;
    use logos::Logos;

    #[test]
    fn module() {
        let source = "module top { input a; output b; wire c; }";
        let mut lex = Token::lexer(source);

        while let Some(token) = lex.next() {
            println!("{:?}", token);
        }
    }
}
