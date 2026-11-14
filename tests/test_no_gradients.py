"""The project's rule: nothing is trained. No backward pass, no optimiser and no gradient, in the package or in its scripts."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRADIENT_CODE = re.compile(
    r"\.backward\(|enable_grad\(|set_grad_enabled\(\s*True|torch\.optim|\boptim\.[A-Z]|requires_grad\s*=\s*True|requires_grad_\(\s*(True)?\s*\)|torch\.autograd|autograd\.grad|\.grad\b"
)


def test_no_gradient_code_in_the_package_or_the_scripts():
    found = []
    for folder in ("src", "scripts"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if GRADIENT_CODE.search(line):
                    found.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:100]}")
    assert not found, "gradient code found:\n" + "\n".join(found)


def test_the_check_itself_recognises_gradient_code():
    for line in ("loss.backward()", "with torch.enable_grad():", "optimizer = torch.optim.Adam([delta], lr=0.1)", "delta = torch.zeros(4, requires_grad=True)",
                 "p.requires_grad_()", "g = torch.autograd.grad(y, x)", "table = probe.grad[0]"):
        assert GRADIENT_CODE.search(line), line
    for line in ("p.requires_grad_(False)", "p.requires_grad_(flag)", "with torch.no_grad():", "left = torch.linalg.solve(gram, right)", "a gradient is out of scope"):
        assert not GRADIENT_CODE.search(line), line
