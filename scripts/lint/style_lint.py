#!/usr/bin/env python3
"""
Lightweight style linter for mlpack C++ sources.

Deliberately narrow in scope. The linter fixes only mechanical,
unambiguous things (trailing whitespace, tabs, `if(` spacing, missing
EOF newline, CRLF) and reports the judgment calls (long lines, brace
placement) so the maintainer stays in control of template heavy code
that clang-format butchers.

Auto-fixes only touch code regions. Strings, character literals, raw
strings, and comments are excluded via a single-pass C++ tokenizer.

Usage:
    scripts/style_lint.py [--fix | --check] PATH [PATH ...]
"""

from __future__ import annotations

import argparse
import codecs
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterable

SOURCE_EXTS = (".hpp", ".cpp", ".h", ".cc", ".cxx")
MAX_LINE = 80
EXCLUDE_DIRS = {
    ".git",
    "build",
    "_build",
    "third_party",
    "_deps",
    "CMakeFiles",
    "bundled",
    "cereal",
}
EXCLUDE_FILES = {"catch.hpp"}

_CODE = ord("c")
_STR = ord("s")
_COMMENT = ord("/")
_NL = ord("\n")

KEYWORD_PAREN = re.compile(r"\b(if|for|while|switch|catch)\(")
_TEMPLATE_WORD = re.compile(r"\btemplate\b")

FIX_IF_GLUED = re.compile(
    r"^(\s*)((?:if|while|for|switch|catch)\s*\(.*\))\s*\{\s*$"
)
FIX_DO_TRY_GLUED = re.compile(r"^(\s*)(do|try)\s*\{\s*$")
FIX_ELSE_IF_GLUED = re.compile(r"^(\s*)\}\s*(else\s+if\s*\(.*\))\s*\{\s*$")
FIX_ELSE_GLUED = re.compile(r"^(\s*)\}\s*else\s*\{\s*$")
FIX_ELSE_UNBRACED = re.compile(r"^(\s*)\}\s*else\s*$")

TRAILING_BRACE = re.compile(
    r"^\s*(?:\}\s*)?(?:if|else|for|while|switch|catch)\b.*\)\s*\{\s*$"
    r"|^\s*\}\s*else\b.*\{\s*$"
    r"|^\s*(?:do|try)\s*\{\s*$"
)
BRACE_ELSE_SAMELINE = re.compile(r"^\s*\}\s*else\b")


def _is_template_context_line(line: str) -> bool:
    """Used to veto brace-placement fixes in template heavy lines."""
    return bool(_TEMPLATE_WORD.search(line)) or (">::" in line)


@dataclass
class Diag:
    line: int
    col: int
    code: str
    msg: str
    fixable: bool


# ===========================================================================
# Tokenizer
# ===========================================================================

def _scan_quoted(
    text: str, out: bytearray, i: int, n: int, quote: str
) -> int:
    """Mask a char or string literal starting at `i` (positioned on the
    opening quote). Returns the index just past the closing quote, or past
    the terminating newline for an unterminated literal."""
    out[i] = _STR
    j = i + 1
    while j < n:
        ch = text[j]
        if ch == "\\" and j + 1 < n:
            out[j] = _STR
            out[j + 1] = _NL if text[j + 1] == "\n" else _STR
            j += 2
            continue
        if ch == quote:
            out[j] = _STR
            j += 1
            break
        if ch == "\n":
            out[j] = _NL
            j += 1
            break
        out[j] = _STR
        j += 1
    return j


def _is_numeric_separator(text: str, i: int) -> bool:
    """True when `text[i]` is a C++14 digit separator inside a numeric
    pp-token rather than a char-literal opener.

    A pp-number starts with a digit; the separator `'` appears between
    alphanumeric run elements of that token. We walk backwards over the
    alnum/underscore run ending at `i` and check whether it begins with a
    digit. The walk is bounded by the length of a numeric literal, which
    is tiny in practice.
    """
    if i == 0:
        return False
    t = i
    while t > 0 and (text[t - 1].isalnum() or text[t - 1] in ("_", "'")):
        t -= 1
    return t < i and text[t].isdigit()


def classify(text: str) -> str:
    """Return a mask the same length as `text`.

    Each position is one of:
      'c'   code
      's'   string or char literal (including delimiters)
      '/'   comment (including delimiters)
      '\\n' literal newline (preserved so per-line splits align with text)

    Handles line comments, block comments, char literals, string
    literals with backslash escapes, raw string literals (with prefixes
    like u8R/uR/UR/LR), and C++14 digit separators inside numeric
    literals.
    """
    n = len(text)
    out = bytearray([_CODE]) * n
    i = 0

    while i < n:
        c = text[i]

        if c == "\n":
            out[i] = _NL
            i += 1
            continue

        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = i
            while j < n and text[j] != "\n":
                out[j] = _COMMENT
                j += 1
            i = j
            continue

        if c == "/" and i + 1 < n and text[i + 1] == "*":
            out[i] = _COMMENT
            out[i + 1] = _COMMENT
            j = i + 2
            while j < n:
                if j + 1 < n and text[j] == "*" and text[j + 1] == "/":
                    out[j] = _COMMENT
                    out[j + 1] = _COMMENT
                    j += 2
                    break
                out[j] = _NL if text[j] == "\n" else _COMMENT
                j += 1
            i = j
            continue

        if c == "R" and i + 1 < n and text[i + 1] == '"':
            delim_start = i + 2
            delim_end = delim_start
            while (
                delim_end < n
                and text[delim_end] != "("
                and text[delim_end] != "\n"
                and delim_end - delim_start < 16
            ):
                delim_end += 1
            if delim_end < n and text[delim_end] == "(":
                delim = text[delim_start:delim_end]
                terminator = ")" + delim + '"'
                tlen = len(terminator)
                for k in range(i, delim_end + 1):
                    out[k] = _NL if text[k] == "\n" else _STR
                j = delim_end + 1
                while j < n:
                    if text[j : j + tlen] == terminator:
                        for k in range(j, j + tlen):
                            out[k] = _STR
                        j += tlen
                        break
                    out[j] = _NL if text[j] == "\n" else _STR
                    j += 1
                i = j
                continue

        if c == "'":
            if _is_numeric_separator(text, i):
                i += 1
                continue
            i = _scan_quoted(text, out, i, n, "'")
            continue

        if c == '"':
            i = _scan_quoted(text, out, i, n, '"')
            continue

        i += 1

    return out.decode("latin-1")


# ===========================================================================
# Linter
# ===========================================================================

def _is_code_span(mask_line: str, start: int, end: int) -> bool:
    return mask_line.count("c", start, end) == end - start


# Table of brace-fix rules. Each entry supplies a pattern, the diagnostic
# code and message, and a function that turns a regex Match into the list
# of replacement lines to emit in place of the original.
_BraceSplit = Callable[["re.Match[str]"], "list[str]"]
_BRACE_FIX_RULES: "list[tuple[re.Pattern[str], str, str, _BraceSplit]]" = [
    (
        FIX_IF_GLUED,
        "brace-same-line",
        "opening brace should be on its own line",
        lambda m: [m.group(1) + m.group(2), m.group(1) + "{"],
    ),
    (
        FIX_DO_TRY_GLUED,
        "brace-same-line",
        "opening brace should be on its own line",
        lambda m: [m.group(1) + m.group(2), m.group(1) + "{"],
    ),
    (
        FIX_ELSE_IF_GLUED,
        "brace-else",
        "`} else if` should be split across lines",
        lambda m: [m.group(1) + "}", m.group(1) + m.group(2), m.group(1) + "{"],
    ),
    (
        FIX_ELSE_GLUED,
        "brace-else",
        "`} else {` should be split across lines",
        lambda m: [m.group(1) + "}", m.group(1) + "else", m.group(1) + "{"],
    ),
    (
        FIX_ELSE_UNBRACED,
        "brace-else",
        "`} else` should be split across lines",
        lambda m: [m.group(1) + "}", m.group(1) + "else"],
    ),
]


def lint_text(text: str, fix: bool) -> "tuple[str, list[Diag]]":
    """Return (new_text, diagnostics). new_text == text unless fix=True."""
    diags: list[Diag] = []

    if "\r\n" in text:
        if fix:
            text = text.replace("\r\n", "\n")
        diags.append(Diag(1, 1, "crlf", "file has CRLF line endings", True))

    mask = classify(text)
    text_lines = text.split("\n")
    mask_lines = mask.split("\n")

    trailing_newline = len(text_lines) > 0 and text_lines[-1] == ""
    if trailing_newline:
        text_lines = text_lines[:-1]
        mask_lines = mask_lines[:-1]
    else:
        diags.append(
            Diag(
                max(1, len(text_lines)),
                1,
                "eof-newline",
                "missing newline at end of file",
                True,
            )
        )

    new_body: list[str] = []
    blank_run = 0
    template_depth = 0

    for i, (raw, m) in enumerate(zip(text_lines, mask_lines), start=1):
        line = raw
        line_mask = m
        start_template_depth = template_depth

        if "\t" in line:
            col = line.find("\t")
            if col < len(line_mask) and line_mask[col] == "c":
                diags.append(Diag(i, col + 1, "tab", "tab character", True))
                if fix:
                    new_chars: list[str] = []
                    new_mask_chars: list[str] = []
                    visual = 0
                    for k, ch in enumerate(line):
                        mch = line_mask[k] if k < len(line_mask) else "c"
                        if ch == "\t" and mch == "c":
                            pad = 2 - (visual % 2)
                            new_chars.append(" " * pad)
                            new_mask_chars.append("c" * pad)
                            visual += pad
                        else:
                            new_chars.append(ch)
                            new_mask_chars.append(mch)
                            visual += 1
                    line = "".join(new_chars)
                    line_mask = "".join(new_mask_chars)

        stripped = line.rstrip(" \t")
        if stripped != line:
            trim_start = len(stripped)
            if _is_code_span(line_mask, trim_start, len(line)):
                diags.append(
                    Diag(i, trim_start + 1, "trailing-ws", "trailing whitespace", True)
                )
                if fix:
                    line = stripped
                    line_mask = line_mask[:trim_start]

        if "(" in line:
            kw_hits = [
                m_kw for m_kw in KEYWORD_PAREN.finditer(line)
                if _is_code_span(line_mask, m_kw.start(), m_kw.end())
            ]
        else:
            kw_hits = []
        for m_kw in kw_hits:
            diags.append(
                Diag(
                    i,
                    m_kw.start() + 1,
                    "kw-paren",
                    f"missing space after `{m_kw.group(1)}`",
                    True,
                )
            )
        if fix and kw_hits:
            pieces: list[str] = []
            last = 0
            for m_kw in kw_hits:
                s, e = m_kw.start(), m_kw.end()
                pieces.append(line[last:s])
                pieces.append(m_kw.group(1) + " (")
                last = e
            pieces.append(line[last:])
            line = "".join(pieces)

        max_depth_during = template_depth
        if "<" in line or ">" in line:
            k = 0
            L = len(line)
            while k < L:
                ch = line[k]
                if k < len(line_mask) and line_mask[k] != "c":
                    k += 1
                    continue
                if ch == "<":
                    if k + 1 < L and line[k + 1] in "<=":
                        k += 2
                        continue
                    template_depth += 1
                    if template_depth > max_depth_during:
                        max_depth_during = template_depth
                    k += 1
                    continue
                if ch == ">":
                    if k + 1 < L and line[k + 1] == "=":
                        k += 2
                        continue
                    if k + 1 < L and line[k + 1] == ">":
                        if template_depth >= 2:
                            template_depth -= 2
                        k += 2
                        continue
                    if template_depth > 0:
                        template_depth -= 1
                    k += 1
                    continue
                k += 1

        stripped_end = line.rstrip()
        ends_with_continuation = bool(stripped_end) and stripped_end.endswith(
            (",", "\\")
        )
        # A single-line `<` (comparison operator) that leaves
        # max_depth_during > 0 must NOT suppress long-line on its own line.
        # Only treat max_depth_during as evidence of template context when
        # the line looks like an opener of a multi-line parameter list,
        # i.e. it ends in a continuation token.
        in_template_context = (
            start_template_depth > 0
            or (max_depth_during > 0 and ends_with_continuation)
            or _TEMPLATE_WORD.search(line) is not None
        )
        if stripped_end and not ends_with_continuation:
            template_depth = 0

        visual_len = len(line.expandtabs(2))
        if visual_len > MAX_LINE and not in_template_context:
            ls = line.lstrip()
            if not (ls.startswith("#include") or "://" in line):
                diags.append(
                    Diag(
                        i,
                        MAX_LINE + 1,
                        "long-line",
                        f"line is {visual_len} > {MAX_LINE} cols",
                        False,
                    )
                )

        fixed_lines: "list[str] | None" = None
        if not _is_template_context_line(line):
            for rgx, code, msg, split in _BRACE_FIX_RULES:
                match = rgx.match(line)
                if match:
                    diags.append(Diag(i, 1, code, msg, True))
                    if fix:
                        fixed_lines = split(match)
                    break
        else:
            if TRAILING_BRACE.match(line):
                diags.append(
                    Diag(i, 1, "brace-same-line", "opening brace should be on its own line", False)
                )
            elif BRACE_ELSE_SAMELINE.match(line):
                diags.append(
                    Diag(i, 1, "brace-else", "`} else` should be split across lines", False)
                )

        emit_lines = fixed_lines if fixed_lines is not None else [line]

        for el in emit_lines:
            if not el or el.isspace():
                blank_run += 1
                if blank_run == 3:
                    diags.append(
                        Diag(i, 1, "blank-run", "3+ consecutive blank lines", True)
                    )
                if fix and blank_run >= 3:
                    continue
            else:
                blank_run = 0
            new_body.append(el)

    new_text = "\n".join(new_body)
    if new_body and (trailing_newline or fix):
        new_text += "\n"

    return (new_text if fix else text), diags


# ===========================================================================
# Atomic file writes
# ===========================================================================

def atomic_write(path: str, content: str) -> None:
    """Write `content` to `path` atomically via tempfile + os.replace.

    Preserves the target file's mode bits on POSIX. Cleans up the temp
    file on any exception, including KeyboardInterrupt, which is why the
    `except` catches BaseException.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        original_mode = os.stat(path).st_mode & 0o7777
    except FileNotFoundError:
        original_mode = None

    fd, tmp_path = tempfile.mkstemp(
        prefix=".style_lint.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        if original_mode is not None:
            try:
                os.chmod(tmp_path, original_mode)
            except OSError:
                pass
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ===========================================================================
# File I/O: encoding detection and full file processing
# ===========================================================================

def read_source(path: str) -> "tuple[str, str]":
    """Decode `path` and return (text, encoding_name).

    Returned encoding is one of 'utf-16', 'utf-8-sig', 'utf-8'. Raises
    UnicodeDecodeError on unsupported encodings.
    """
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return raw.decode("utf-16"), "utf-16"
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    return raw.decode("utf-8"), "utf-8"


def process_file(path: str, fix: bool) -> "list[Diag]":
    """Read, lint, and (in fix mode) rewrite a single file."""
    try:
        text, encoding = read_source(path)
    except (OSError, UnicodeDecodeError) as e:
        print(f"{path}: skipped ({e})", file=sys.stderr)
        return []

    diags: list[Diag] = []
    if encoding != "utf-8":
        diags.append(
            Diag(1, 1, "encoding", f"file is {encoding}, expected utf-8", True)
        )

    new_text, lint_diags = lint_text(text, fix)
    diags.extend(lint_diags)

    if fix and (new_text != text or encoding != "utf-8"):
        atomic_write(path, new_text)

    return diags


# ===========================================================================
# File discovery + CLI
# ===========================================================================

def iter_sources(paths: Iterable[str]) -> Iterable[str]:
    for p in paths:
        if os.path.isfile(p):
            yield p
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for f in files:
                    if f in EXCLUDE_FILES:
                        continue
                    if f.endswith(SOURCE_EXTS):
                        yield os.path.join(root, f)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Lightweight style linter for mlpack C++ sources."
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--fix", action="store_true", help="apply safe rewrites in place"
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="report-only; exit 1 only if a fixable issue exists (CI mode)",
    )
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    fixable_total = 0
    other_total = 0
    for path in iter_sources(args.paths):
        diags = process_file(path, fix=args.fix)
        for d in diags:
            if args.fix and d.fixable:
                continue
            print(f"{path}:{d.line}:{d.col}: {d.code}: {d.msg}")
        fixable_total += sum(1 for d in diags if d.fixable)
        other_total += sum(1 for d in diags if not d.fixable)

    if args.fix:
        return 1 if other_total else 0
    if args.check:
        return 1 if fixable_total else 0
    return 1 if (fixable_total + other_total) else 0


if __name__ == "__main__":
    sys.exit(main())
