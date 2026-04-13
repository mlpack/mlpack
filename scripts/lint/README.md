# `scripts/lint/`: mlpack style linter

A lightweight, tokenizer aware C++ style linter. It exists because
`clang-format` does not deal well with mlpack's heavy template code.
`clang-format` wants to reformat the entire file, and its idea of how to
wrap template argument lists disagrees with the maintainers'. This
linter takes a narrower contract: **fix the mechanical stuff, leave the
rest to humans.**

## What it does

Mechanical auto-fixes (safe to apply without review):

| Code | What | Example |
|---|---|---|
| `encoding` | UTF-16 or UTF-8 with BOM rewritten as UTF-8 | Visual Studio sample sources |
| `crlf` | CRLF line endings rewritten as LF | Files committed from Windows |
| `tab` | tabs in code rewritten as 2 spaces | (string literals left alone) |
| `trailing-ws` | trailing whitespace on code lines | (raw-string content left alone) |
| `eof-newline` | missing final newline | |
| `kw-paren` | `if(`, `for(`, `while(`, `switch(`, `catch(` rewritten as `if (` etc. | |
| `brace-same-line` | `if (x) {` split into `if (x)` then `{` on the next line | Allman style |
| `brace-else` | `} else {` split into `}`, `else`, `{` on three lines | |
| `blank-run` | 3+ consecutive blank lines collapsed to 2 | |

Report only (flagged for human judgment):

| Code | What |
|---|---|
| `long-line` | lines > 80 cols, **suppressed inside template context** |

Long lines are deliberately not auto-wrapped. That is where
`clang-format` gets it wrong on template code, and mlpack would rather
have the maintainer pick the wrap point.

## Running it

```bash
# Report mode. Prints every diagnostic and exits non-zero if anything
# was found, fixable or not.
python3 scripts/lint/style_lint.py src/

# Check mode. What CI uses. Prints every diagnostic, but exits
# non-zero *only* if something auto-fixable exists. Baseline long-line
# warnings do not trip it.
python3 scripts/lint/style_lint.py --check src/

# Fix mode. Applies safe rewrites atomically (via tempfile plus
# os.replace) and reports whatever non-fixable remains for human review.
python3 scripts/lint/style_lint.py --fix src/

# Whole repo:
python3 scripts/lint/style_lint.py --check .
```

The three modes are mutually exclusive. The common workflow is:
`--check` locally before pushing, `--fix` when you want the rewrites
applied, and plain report when you want to see everything including
non-fixable diagnostics.

Runtime on the full mlpack tree (~1550 C++ files) is around 2 seconds.
Fix mode is essentially the same speed, because the rewrite path is
only hit for the handful of files that actually need changes.

## Safety: the tokenizer

The linter contains a single-pass C++ tokenizer (`classify()`) that
produces a mask over every byte of the file:

* `c` for code
* `s` for string or char literal (including delimiters)
* `/` for comment (including delimiters)
* `\n` preserved so per-line splits line up with the text

Every auto-fix checks the mask before rewriting. The consequence is
that things like

```cpp
const char* msg = "if(x) failed";
const char* raw = R"(if(x) { while(true); })";
// uses for(int i=0; i<n; ++i) internally
/* if(y) {} */
```

pass through `--fix` completely untouched, while the same constructs
outside strings and comments get rewritten. The tokenizer understands
line comments, block comments (including across newlines), char
literals, strings with backslash escapes, and raw string literals with
prefixes (`u8R`, `uR`, `UR`, `LR`) and custom delimiters up to 16
characters.

## Exclusions

Directories never walked:

```
.git/  build/  _build/  third_party/  _deps/  CMakeFiles/  bundled/  cereal/
```

Files excluded by name:

```
catch.hpp
```

Rationale: `bundled/`, `cereal/`, and `catch.hpp` are vendored
third-party sources. We do not want to retrofit their upstream style,
and drive-by rewrites make future upstream syncs painful.

## Template aware long-line suppression

The long-line check tracks angle-bracket depth across lines, in code
regions only. When the depth is > 0, or the line contains the word
`template`, long-line warnings are suppressed. The depth is clamped
back to 0 at every line that does not end in `,` or `\`, so that a
`for (int i = 0; i < n; ++i)` comparison operator does not poison the
rest of the file.

This heuristic is imperfect by design. The alternative (a full C++
parser) is miles away from the effort we want to spend here. If it
occasionally under-reports a long line inside complex template code,
that is fine. Complex template code is where `clang-format` was wrong
anyway.

## Extending the linter

### Adding a new auto-fix

1. **Write the test first.** Use `scripts/lint/test_style_lint.py` as a
   template. Include at least: a happy path fix case, a case that must
   be skipped inside a string literal, and a case that must be skipped
   inside a comment. Run the tests and watch them fail.
2. **Add the rule to `lint_text()`**, not outside it. Rules are applied
   in line order so they can see each other's output via `line_mask`.
3. **Use the mask.** For regex-based rewrites, wrap every match in a
   `_is_code_span(line_mask, start, end)` check before applying it.
   For character-level rewrites (like tab expansion), check
   `line_mask[k] == "c"` per character.
4. **Emit a `Diag`** with `fixable=True` whenever you would rewrite.
   The Diag itself is reported unconditionally; the `fixable` flag
   tells `main` whether to suppress it after a successful `--fix`.
5. **Atomic write is handled for you** via `atomic_write()` in
   `process_file()`. You do not need to touch file I/O.

### Adding a new report-only rule

Same as above, but create the Diag with `fixable=False`. The rule can
live in `lint_text()` next to the auto-fix rules; the only difference
is that no rewrite happens.

### Adding an exclusion

Add the directory name to `EXCLUDE_DIRS` or the file basename to
`EXCLUDE_FILES`. These are matched against the bare directory or file
name during `os.walk`, not against full paths. If you need a
path-prefix exclusion, extend `iter_sources()`.

### Things to be careful about

* **Do not rewrite inside `s` or `/` mask regions.** That is the whole
  point of the tokenizer. A rewrite that corrupts a string literal is
  much worse than any style nit it could fix.
* **Assume the line may change under your feet.** Rules run in a fixed
  order inside `lint_text()`. If a rule rewrites the line, subsequent
  rules see the new content, and `line_mask` is kept in sync by each
  fixing rule. If you add a rule that changes line length, update
  `line_mask` the same way.
* **Brace fix style rules may emit multiple output lines** for one
  input line. Follow the pattern in `lint_text()`: produce a
  `fixed_lines: list[str] | None` and let the trailing emission loop
  do the work.
* **Do not touch the long-line heuristic lightly.** The template depth
  tracking looks trivial but has sharp edges around `<<`, `>>`, `<=`,
  `>=`, and comparison operators. There are regression tests for all
  of these; make sure they still pass.
* **Never bypass `atomic_write()` for in-place edits.** An interrupted
  rewrite of a header file in the middle of a large run is worse than
  any bug the linter could catch.

## Tests

Pure stdlib unittest. No pytest, no third-party dependencies.

```bash
python3 scripts/lint/test_style_lint.py
```

The test suite covers the tokenizer (string, comment, and raw-string
edge cases), every auto-fix rule, template-context long-line
suppression, comparison operator non-poisoning, exclusion rules,
encoding detection, atomic writes, and the `--check` exit code
contract. The full suite runs in under 100 ms.

Required: every new rule ships with tests. The review rule is simple.
If you did not watch a failing test for your rule, it did not exist
first, and you cannot prove the rule actually tests anything.

## CI

See `.github/workflows/style-lint.yml`. The workflow runs
`scripts/lint/test_style_lint.py` to sanity check the linter itself,
then runs `scripts/lint/style_lint.py --check .` on the full tree.
Both must exit zero for the job to pass.

`--check` rather than plain report mode is intentional. Baseline
long-line warnings in the existing tree would break CI on every PR
otherwise. `--check` surfaces them in the log so contributors can
still see them, but it only fails the job if something auto-fixable
exists.

Auto-fix is **not** run in CI. Rewriting files during CI is something
you do locally, review in a PR, and commit deliberately.
