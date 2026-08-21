from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_IMPORTS = {"requests", "boto3", "openai", "anthropic", "google.generativeai"}


def main() -> None:
    violations: list[str] = []
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module}
            else:
                continue
            if any(
                any(name == item or name.startswith(item + ".") for item in FORBIDDEN_IMPORTS)
                for name in names
            ):
                violations.append(f"{path}:{node.lineno}")
    if violations:
        raise SystemExit("Forbidden network-client imports: " + ", ".join(violations))
    print("Offline import policy check passed.")


if __name__ == "__main__":
    main()
