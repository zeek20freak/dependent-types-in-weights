"""REPL playground for the v0.1 dependent kernel (rung 1).

Run it:  python3 repl_dt.py

Type a term and get its type back, or a term and a type and get the
checker's verdict. The kernel is the judge — this file is only the
front door: a small parser for surface syntax, plus a read-eval-print
loop. The parser is NOT part of the sealed kernel; it is verified
against it in test_repl.py, which parses every example from
examples_dt.py in surface syntax and checks the verdicts match.

Surface syntax:
    \\x:A. body        a function (backslash or the letter lambda)
    Pi x:A. body       a dependent function type (Pi, or the letter Pi)
    A -> B             a non-dependent function type (or the arrow)
    Type 0, Type 1     universes
    f x                application, by putting terms next to each other
    (t)                grouping
Names are letters, digits, underscores, primes: A, B, x, f, A'.

In the loop:
    <term>                 infer the term's type
    <term> : <type>        check the term against the type
    :defeq <t> == <u>      test definitional equality
    :decl <name> : <type>  add a name to the context
    :ctx                  show the context
    :help                 this list
    :quit                 leave
Lines starting with # are comments. The context starts with
A, B, C : Type 0 — the same declared base types as examples_dt.py.
"""

from kernel_dt import (Var, Lam, App, Pi, Type, check_or_fail,
                       infer_or_fail, defeq, _level_of_type, CheckError)

ctx0 = {"A": Type(0), "B": Type(0), "C": Type(0)}


class ParseError(Exception):
    pass


# --- Tokenizer -------------------------------------------------------------

def tokenize(src):
    """Split surface syntax into (kind, value) tokens."""
    toks = []
    i = 0
    while i < len(src):
        ch = src[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "#":  # comment to end of line
            break
        if ch in "\\\u03bb":  # backslash or lambda
            toks.append(("LAM", ch))
            i += 1
            continue
        if ch == "\u03a0":  # capital Pi
            toks.append(("PI", ch))
            i += 1
            continue
        if ch == "\u2192":  # rightwards arrow
            toks.append(("ARROW", ch))
            i += 1
            continue
        if ch == ".":
            toks.append(("DOT", ch))
            i += 1
            continue
        if ch == ":":
            toks.append(("COLON", ch))
            i += 1
            continue
        if ch == "(":
            toks.append(("LPAREN", ch))
            i += 1
            continue
        if ch == ")":
            toks.append(("RPAREN", ch))
            i += 1
            continue
        if src.startswith("->", i):
            toks.append(("ARROW", "->"))
            i += 2
            continue
        if src.startswith("==", i):
            toks.append(("EQ", "=="))
            i += 2
            continue
        if ch.isdigit():
            j = i
            while j < len(src) and src[j].isdigit():
                j += 1
            toks.append(("INT", int(src[i:j])))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < len(src) and (src[j].isalnum() or src[j] in "_'"):
                j += 1
            word = src[i:j]
            if word == "Pi":
                toks.append(("PI", word))
            elif word == "Type":
                toks.append(("TYPE", word))
            else:
                toks.append(("IDENT", word))
            i = j
            continue
        raise ParseError(f"unexpected character {ch!r} at position {i}")
    toks.append(("EOF", None))
    return toks


# --- Parser -----------------------------------------------------------------
# Grammar (lambda and Pi bodies extend as far right as possible;
# application binds tightest; -> is right-associative):
#   term  := lam | pi | arrow
#   lam   := ('\\' | lambda) IDENT ':' term '.' term
#   pi    := ('Pi' | Pi) IDENT ':' term '.' term
#   arrow := app ('->' arrow)?
#   app   := atom+
#   atom  := '(' term ')' | 'Type' INT | IDENT

class Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i]

    def next(self):
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def expect(self, kind):
        tok = self.next()
        if tok[0] != kind:
            raise ParseError(f"expected {kind}, got {tok[0]}"
                             + (f" ({tok[1]!r})" if tok[1] is not None
                                else ""))
        return tok

    def parse_term(self):
        kind, _ = self.peek()
        if kind == "LAM":
            self.next()
            _, var = self.expect("IDENT")
            self.expect("COLON")
            var_type = self.parse_term()
            self.expect("DOT")
            body = self.parse_term()
            return Lam(var, var_type, body)
        if kind == "PI":
            self.next()
            _, var = self.expect("IDENT")
            self.expect("COLON")
            var_type = self.parse_term()
            self.expect("DOT")
            body = self.parse_term()
            return Pi(var, var_type, body)
        return self.parse_arrow()

    def parse_arrow(self):
        left = self.parse_app()
        if self.peek()[0] == "ARROW":
            self.next()
            right = self.parse_arrow()
            return Pi("_", left, right)
        return left

    def parse_app(self):
        func = self.parse_atom()
        while self.peek()[0] in ("LPAREN", "IDENT", "TYPE"):
            func = App(func, self.parse_atom())
        return func

    def parse_atom(self):
        kind, val = self.peek()
        if kind == "LPAREN":
            self.next()
            term = self.parse_term()
            self.expect("RPAREN")
            return term
        if kind == "TYPE":
            self.next()
            _, level = self.expect("INT")
            return Type(level)
        if kind == "IDENT":
            self.next()
            return Var(val)
        raise ParseError(f"expected a term, got {kind}"
                         + (f" ({val!r})" if val is not None else ""))


def parse_term_source(src):
    """Parse one term from surface syntax."""
    parser = Parser(tokenize(src))
    term = parser.parse_term()
    parser.expect("EOF")
    return term


def parse_line(src):
    """Parse a REPL line: a term, optionally followed by : and a type."""
    parser = Parser(tokenize(src))
    term = parser.parse_term()
    typ = None
    if parser.peek()[0] == "COLON":
        parser.next()
        typ = parser.parse_term()
    parser.expect("EOF")
    return term, typ


# --- The loop ---------------------------------------------------------------

HELP = """Commands:
  <term>                 infer the term's type
  <term> : <type>        check the term against the type
  :defeq <t> == <u>      test definitional equality
  :decl <name> : <type>  add a name to the context
  :ctx                   show the context
  :help                  this list
  :quit                  leave
Terms: \\x:A. body   Pi x:A. body   A -> B   Type 0   f x   (t)"""


def handle_command(line, ctx):
    """Run one :command. Returns True when the loop should stop."""
    parts = line.split(None, 1)
    cmd = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if cmd in (":quit", ":q"):
        return True
    if cmd == ":help":
        print(HELP)
    elif cmd == ":ctx":
        for name, typ in ctx.items():
            print(f"  {name} : {typ}")
    elif cmd == ":decl":
        try:
            parser = Parser(tokenize(rest))
            _, name = parser.expect("IDENT")
            parser.expect("COLON")
            typ = parser.parse_term()
            parser.expect("EOF")
        except ParseError as e:
            print(f"parse error: {e}")
            return False
        try:
            _level_of_type(ctx, typ)
        except CheckError as e:
            print(f"not a type: {e}")
            return False
        ctx[name] = typ
        print(f"{name} : {typ}")
    elif cmd == ":defeq":
        try:
            parser = Parser(tokenize(rest))
            t = parser.parse_term()
            parser.expect("EQ")
            u = parser.parse_term()
            parser.expect("EOF")
        except ParseError as e:
            print(f"parse error: {e}")
            return False
        try:
            print("defeq =", defeq(ctx, t, u))
        except CheckError as e:
            print(f"defeq error: {e}")
    else:
        print(f"unknown command {cmd} — :help lists the commands")
    return False


def repl():
    ctx = dict(ctx0)
    print("Dependent kernel REPL — rung 1 (Pi + universes + conversion).")
    print("Type a term to infer its type, or <term> : <type> to check it.")
    print("Commands: :help :ctx :decl :defeq :quit")
    while True:
        try:
            line = input("dt> ")
        except EOFError:
            print()
            break
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(":"):
            if handle_command(line, ctx):
                break
            continue
        try:
            term, typ = parse_line(line)
        except ParseError as e:
            print(f"parse error: {e}")
            continue
        if typ is None:
            inferred, fail = infer_or_fail(ctx, term)
            if fail is None:
                print(f": {inferred}")
            else:
                print(f"REJECT at {fail['path']}, {fail['kind']}: "
                      f"{fail['message']}")
        else:
            fail = check_or_fail(ctx, term, typ)
            if fail is None:
                print("ACCEPT")
            else:
                print(f"REJECT at {fail['path']}, {fail['kind']}: "
                      f"{fail['message']}")


def main():
    repl()


if __name__ == "__main__":
    main()
