import ast
import pathlib
import sys


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body.pop(0)
            if not body:
                body.append(ast.Pass())
    return ast.fix_missing_locations(tree)


for root in sys.argv[1:]:
    for p in pathlib.Path(root).rglob("*.py"):
        tree = strip_docstrings(ast.parse(p.read_text(encoding="utf-8")))
        p.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
    for p in pathlib.Path(root).rglob("*.yaml"):
        lines = []
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.lstrip().startswith("#"):
                continue
            if " #" in l and '"' not in l and "'" not in l:
                l = l[: l.index(" #")].rstrip()
            if l.strip():
                lines.append(l)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
