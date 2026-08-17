#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Enforce the direction of dependence: nothing shippable may reach outside.

lgpsf-hessian is a general-purpose library; the glaciology work that produced
it is a downstream consumer.  So the ice-sheet code may depend on
lgpsf-hessian, and lgpsf-hessian may not depend on it -- nor on any other
private repository or absolute path that exists on one machine.

This is the mechanical form of that rule (adapted from lgpsf's checker, with
the lesson learned there applied: CLAUDE.md and README are in the shipped
set, because they are tracked and public even when excluded from tarballs).

Naming a private problem in prose is FINE and deliberate ("validated on the
Pine Island Glacier Hessian").  What is banned is a path: an include, a
build flag, or a filename that only exists on one machine.

Also enforced: the public C headers (include/lgpsf_hessian/*.h) must stay
dependency-light -- no lgpsf, Eigen, or C++ includes; heavyweight includes
belong in include/lgpsf_hessian/impl/ and impl.hpp.

Run:  python tools/check_dependencies.py
"""
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# Everything tracked that a reader can reach.  (tools/ is exempt: this file
# necessarily contains the banned patterns themselves.)
SHIPPED = ["include", "tests", "examples", "docs", "cmake", ".github",
           "CMakeLists.txt", "README.md", "CLAUDE.md", "LICENSE"]

# Public libraries we explicitly depend on; referencing them is correct.
ALLOWED_EXTERNAL = {"lgpsf", "ellipsoid_tree", "lgpsf_hessian"}

PATTERNS = [
    (re.compile(r"(?:~|/home/[^/\s]+|/Users/[^/\s]+)/"),
     "an absolute path that exists on one machine"),
    (re.compile(r"\b(?:nicks_research_experiments|ellipsoid_psf\w*"
                r"|localpsf\w*|ymir[\w-]*)/"),
     "a path into a private repo"),
    (re.compile(r"\bslice\d+[a-z]?_[a-z0-9_]+\.(?:py|npz|cpp)\b"),
     "a file that only exists in the private research repo"),
]

SKIP_SUFFIXES = {".png", ".pdf", ".pyc", ".so", ".o"}

# Public C headers must not pull in the heavyweight C++ stack.
PUBLIC_HEADER_BANNED_INCLUDE = re.compile(
    r'#include\s+[<"](?:lgpsf/|Eigen|eigen|unsupported/)')


def public_header_problems():
    root = REPO / "include" / "lgpsf_hessian"
    if not root.exists():
        return
    for path in sorted(root.glob("*.h")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if PUBLIC_HEADER_BANNED_INCLUDE.search(line):
                yield (f"include/lgpsf_hessian/{path.name}:{number}: a public "
                       f"C header includes the C++ stack\n    {line.strip()}")


def offending_lines(path):
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    for number, line in enumerate(text.splitlines(), start=1):
        for pattern, why in PATTERNS:
            match = pattern.search(line)
            if match and match.group(0).strip("/ ") not in ALLOWED_EXTERNAL:
                yield number, line.strip(), why


def main():
    problems = []
    for entry in SHIPPED:
        root = REPO / entry
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix in SKIP_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            for number, line, why in offending_lines(path):
                problems.append(
                    f"{path.relative_to(REPO)}:{number}: {why}\n    {line}")

    if problems:
        print("lgpsf-hessian must not depend on anything private. Found "
              f"{len(problems)} reference(s):\n", file=sys.stderr)
        for problem in problems:
            print(problem, file=sys.stderr)
        print("\nNaming a private problem in PROSE is fine. A PATH is not.",
              file=sys.stderr)
        return 1

    crossings = list(public_header_problems())
    if crossings:
        print("Public C headers must stay dependency-light; found "
              f"{len(crossings)} heavyweight include(s):\n", file=sys.stderr)
        for crossing in crossings:
            print(crossing, file=sys.stderr)
        return 1

    print(f"ok: no private dependencies in {', '.join(SHIPPED)}; "
          "public headers stay dependency-light")
    return 0


if __name__ == "__main__":
    sys.exit(main())
