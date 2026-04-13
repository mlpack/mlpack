"""Tests for scripts/style_lint.py.

`classify(text)` should return a mask string the same length as `text`:
  'c' = code, 's' = string or char literal, '/' = comment.

`lint_text(text, fix=True)` must only auto-fix inside code regions.

`atomic_write(path, content)` writes via a temp file + os.replace.
"""

import os
import stat
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_lint as sl  # noqa: E402


class TestClassify(unittest.TestCase):
    def assertMask(self, text, expected):
        got = sl.classify(text)
        self.assertEqual(
            got, expected, f"\ntext:     {text!r}\ngot:      {got}\nexpected: {expected}"
        )

    def test_plain_code(self):
        self.assertMask("if (x)", "cccccc")

    def test_simple_string(self):
        self.assertMask('"abc"', "sssss")

    def test_code_and_string(self):
        self.assertMask('x="ab"', "ccssss")

    def test_line_comment(self):
        self.assertMask("x//y", "c///")

    def test_line_comment_ends_at_newline(self):
        self.assertMask("x//y\nz", "c///\nc")

    def test_block_comment_single_line(self):
        self.assertMask("/*x*/y", "/////c")

    def test_block_comment_multiline(self):
        self.assertMask("a/*\nx\n*/b", "c//\n/\n//c")

    def test_char_literal(self):
        self.assertMask("'a'", "sss")

    def test_escaped_quote_in_string(self):
        # "a\"b" -> 6 chars, all string
        self.assertMask('"a\\"b"', "ssssss")

    def test_slashslash_inside_string_is_not_comment(self):
        self.assertMask('"//"', "ssss")

    def test_star_slash_inside_string_does_not_end_block_comment(self):
        # Start outside; string contains "*/" which must not be seen as end.
        self.assertMask('x="*/";', "ccssssc")

    def test_raw_string_simple(self):
        # R"(ab)"
        self.assertMask('R"(ab)"', "sssssss")

    def test_raw_string_multiline(self):
        # R"(a\nb)". The newline inside the raw string stays as a newline in
        # the mask so per-line splits line up; surrounding chars are string.
        self.assertMask('R"(a\nb)"', "ssss\nsss")

    def test_raw_string_with_delimiter(self):
        # R"xx(a)"b)xx"
        text = 'R"xx(a)"b)xx"'
        self.assertMask(text, "s" * len(text))

    def test_raw_string_slash_slash_inside(self):
        # `//` inside raw string is content, not comment.
        self.assertMask('R"(//)"', "sssssss")

    def test_u8_raw_string(self):
        # u8R"(a)". The u8 prefix is classified as code, which is safe
        # because nothing we rewrite matches it. The raw-string body is 's'.
        self.assertMask('u8R"(a)"', "ccssssss")

    def test_block_comment_with_slash_slash_inside(self):
        self.assertMask("/*//*/", "//////")


class TestLintFixesSkipNonCode(unittest.TestCase):
    def test_fix_rewrites_if_paren_in_code(self):
        text = "if(x)\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "if (x)\n")

    def test_fix_does_not_rewrite_if_paren_in_string(self):
        text = 'const char* msg = "if(x) failed";\n'
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, text)

    def test_fix_does_not_rewrite_if_paren_in_line_comment(self):
        text = "int a = 0; // uses if(x) internally\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, text)

    def test_fix_does_not_rewrite_if_paren_in_block_comment(self):
        text = "/* if(x) */ int a = 0;\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, text)

    def test_fix_does_not_rewrite_if_paren_in_raw_string(self):
        text = 'auto s = R"(if(x))";\n'
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, text)

    def test_fix_preserves_tabs_inside_string(self):
        text = 'auto s = "a\tb";\n'
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, text)

    def test_fix_converts_tab_in_code(self):
        # Tab at column 3 expands to the next multiple of 2 (column 4) = 1 space.
        text = "int\ta = 0;\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "int a = 0;\n")

    def test_fix_skips_trailing_ws_inside_raw_string(self):
        # Trailing whitespace on a line whose tail is inside a raw string
        # must not be stripped; it is real string content.
        text = 'auto s = R"(a   \nb)";\n'
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, text)

    def test_fix_strips_trailing_ws_on_code_line(self):
        text = "int a = 0;   \n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "int a = 0;\n")

    def test_fix_adds_eof_newline(self):
        text = "int a = 0;"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "int a = 0;\n")

    def test_fix_normalises_crlf(self):
        text = "int a = 0;\r\nint b = 1;\r\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "int a = 0;\nint b = 1;\n")

    def test_report_mode_does_not_mutate(self):
        text = "if(x)\n"
        new_text, diags = sl.lint_text(text, fix=False)
        self.assertEqual(new_text, text)
        self.assertTrue(any(d.code == "kw-paren" for d in diags))


class TestBraceFix(unittest.TestCase):
    def test_fix_if_glued_brace(self):
        text = "  if (x) {\n    y;\n  }\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "  if (x)\n  {\n    y;\n  }\n")

    def test_fix_for_glued_brace(self):
        text = "  for (int i = 0; i < n; ++i) {\n    y;\n  }\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "  for (int i = 0; i < n; ++i)\n  {\n    y;\n  }\n")

    def test_fix_while_glued_brace(self):
        text = "while (x) {\n}\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "while (x)\n{\n}\n")

    def test_fix_do_glued_brace(self):
        text = "  do {\n    y;\n  } while (x);\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "  do\n  {\n    y;\n  } while (x);\n")

    def test_fix_try_glued_brace(self):
        text = "try {\n}\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "try\n{\n}\n")

    def test_fix_else_glued_brace(self):
        text = "  } else {\n    y;\n  }\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "  }\n  else\n  {\n    y;\n  }\n")

    def test_fix_else_if_glued_brace(self):
        text = "  } else if (x) {\n    y;\n  }\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "  }\n  else if (x)\n  {\n    y;\n  }\n")

    def test_fix_else_without_brace(self):
        # `} else` on its own, followed by a block below, should split `}` off.
        text = "  } else\n  {\n    y;\n  }\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "  }\n  else\n  {\n    y;\n  }\n")

    def test_skip_brace_fix_in_template_context(self):
        # Line contains `template<`: do not touch, reserved for human.
        text = "template<typename T> void Foo() {\n}\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, text)

    def test_skip_brace_fix_with_template_arg(self):
        # `if (std::is_same<A, B>::value) {` has template args in the
        # condition, so leave it alone.
        text = "if (std::is_same<A, B>::value) {\n}\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, text)

    def test_brace_fix_does_not_touch_lambda(self):
        # Lambda `[](int x) { return x; }` is not control flow, so no fix.
        text = "auto f = [](int x) { return x; };\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, text)

    def test_brace_fix_ignores_glued_brace_in_string(self):
        text = 'const char* s = "if (x) {";\n'
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, text)


class TestBlankRunFix(unittest.TestCase):
    def test_collapses_four_blanks_to_two(self):
        text = "a;\n\n\n\n\nb;\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, "a;\n\n\nb;\n")

    def test_two_blanks_untouched(self):
        text = "a;\n\n\nb;\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertEqual(new_text, text)


class TestLongLineTemplateSuppression(unittest.TestCase):
    def _mk_long(self, prefix: str) -> str:
        # Build a line of >80 cols by padding with identifiers/commas.
        line = prefix
        while len(line) <= 80:
            line += "LongNameX, "
        return line.rstrip(", ")

    def test_long_line_flagged_in_plain_code(self):
        line = "int foo = " + "a + " * 25 + "0;"
        text = line + "\n"
        _, diags = sl.lint_text(text, fix=False)
        self.assertTrue(any(d.code == "long-line" for d in diags))

    def test_long_line_suppressed_on_template_keyword_line(self):
        line = "template<typename " + "LongName, typename " * 5 + "Last>"
        self.assertGreater(len(line), 80)
        _, diags = sl.lint_text(line + "\n", fix=False)
        self.assertFalse(any(d.code == "long-line" for d in diags))

    def test_long_line_suppressed_on_template_continuation(self):
        # First line opens template<...> without closing on same line; second
        # line is inside the template parameter list and too long.
        text = (
            "template<typename A,\n"
            "         typename " + "LongName" * 10 + ">\n"
            "void foo();\n"
        )
        _, diags = sl.lint_text(text, fix=False)
        long_diags = [d for d in diags if d.code == "long-line"]
        self.assertEqual(long_diags, [])

    def test_comparison_op_does_not_poison_subsequent_lines(self):
        # `<` as a less-than operator must not leave template_depth high
        # for the rest of the file. Subsequent plain long lines must still
        # be reported.
        long_line = "int foo = " + "a + " * 25 + "0;"
        text = "if (x < y) { f(); }\n" + long_line + "\n"
        _, diags = sl.lint_text(text, fix=False)
        self.assertTrue(
            any(d.code == "long-line" for d in diags),
            "long line after an `<` comparison should still be flagged",
        )

    def test_for_loop_comparison_does_not_poison_subsequent_lines(self):
        long_line = "int foo = " + "a + " * 25 + "0;"
        text = "for (int i = 0; i < n; ++i) {}\n" + long_line + "\n"
        _, diags = sl.lint_text(text, fix=False)
        self.assertTrue(any(d.code == "long-line" for d in diags))

    def test_long_line_suppressed_on_template_qualified_name(self):
        # `NeighborSearch<A, B, C,\n` opens angle brackets, so it stays
        # in template territory on the following line.
        text = (
            "NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,\n"
            "    DualTreeTraversalType>::NeighborSearch(int a, int b, int c, int d, int e);\n"
        )
        _, diags = sl.lint_text(text, fix=False)
        long_diags = [d for d in diags if d.code == "long-line"]
        self.assertEqual(long_diags, [])


class TestExclusions(unittest.TestCase):
    def test_bundled_dir_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "a", "bundled"))
            os.makedirs(os.path.join(td, "a", "real"))
            p1 = os.path.join(td, "a", "bundled", "x.hpp")
            p2 = os.path.join(td, "a", "real", "y.hpp")
            for p in (p1, p2):
                with open(p, "w") as f:
                    f.write("int x;\n")
            found = set(sl.iter_sources([td]))
            self.assertIn(p2, found)
            self.assertNotIn(p1, found)

    def test_vendored_files_excluded_by_name(self):
        with tempfile.TemporaryDirectory() as td:
            catch = os.path.join(td, "catch.hpp")
            normal = os.path.join(td, "normal.hpp")
            for p in (catch, normal):
                with open(p, "w") as f:
                    f.write("int x;\n")
            # When passed a directory, catch.hpp should be skipped.
            found = set(sl.iter_sources([td]))
            self.assertIn(normal, found)
            self.assertNotIn(catch, found)

    def test_cereal_dir_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "core", "cereal"))
            p = os.path.join(td, "core", "cereal", "x.hpp")
            with open(p, "w") as f:
                f.write("int x;\n")
            found = set(sl.iter_sources([td]))
            self.assertNotIn(p, found)


class TestReadSource(unittest.TestCase):
    def _write(self, td, name, raw):
        p = os.path.join(td, name)
        with open(p, "wb") as f:
            f.write(raw)
        return p

    def test_plain_utf8(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, "f.hpp", b"int x = 0;\n")
            text, enc = sl.read_source(p)
            self.assertEqual(text, "int x = 0;\n")
            self.assertEqual(enc, "utf-8")

    def test_utf8_with_bom(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, "f.hpp", b"\xef\xbb\xbfint x = 0;\n")
            text, enc = sl.read_source(p)
            self.assertEqual(text, "int x = 0;\n")  # BOM stripped
            self.assertEqual(enc, "utf-8-sig")

    def test_utf16_le_bom(self):
        with tempfile.TemporaryDirectory() as td:
            raw = b"\xff\xfe" + "int x = 0;\n".encode("utf-16-le")
            p = self._write(td, "f.hpp", raw)
            text, enc = sl.read_source(p)
            self.assertEqual(text, "int x = 0;\n")
            self.assertEqual(enc, "utf-16")

    def test_utf16_be_bom(self):
        with tempfile.TemporaryDirectory() as td:
            raw = b"\xfe\xff" + "int x = 0;\n".encode("utf-16-be")
            p = self._write(td, "f.hpp", raw)
            text, enc = sl.read_source(p)
            self.assertEqual(text, "int x = 0;\n")
            self.assertEqual(enc, "utf-16")

    def test_binary_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, "f.hpp", b"\x80\x81\x82\x83 invalid utf-8")
            with self.assertRaises(UnicodeDecodeError):
                sl.read_source(p)


class TestProcessFile(unittest.TestCase):
    def test_fix_rewrites_utf16_as_utf8(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.hpp")
            content = "int x = 0;\n"
            with open(p, "wb") as f:
                f.write(b"\xff\xfe" + content.encode("utf-16-le"))
            diags = sl.process_file(p, fix=True)
            with open(p, "rb") as f:
                raw = f.read()
            self.assertEqual(raw, content.encode("utf-8"))
            self.assertTrue(any(d.code == "encoding" for d in diags))

    def test_fix_strips_utf8_bom(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.hpp")
            with open(p, "wb") as f:
                f.write(b"\xef\xbb\xbfint x = 0;\n")
            sl.process_file(p, fix=True)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"int x = 0;\n")

    def test_report_mode_does_not_rewrite(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.hpp")
            original = b"\xff\xfe" + "int x = 0;\n".encode("utf-16-le")
            with open(p, "wb") as f:
                f.write(original)
            diags = sl.process_file(p, fix=False)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), original)
            self.assertTrue(any(d.code == "encoding" for d in diags))

    def test_utf8_file_untouched_when_no_lint_changes(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.hpp")
            with open(p, "wb") as f:
                f.write(b"int x = 0;\n")
            sl.process_file(p, fix=True)
            # Clean utf-8 file with no diagnostics should NOT be rewritten.
            # (Checking via content is sufficient; mtime is racy.)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"int x = 0;\n")


class TestCheckMode(unittest.TestCase):
    """--check is the CI-facing variant: no rewrites, exit 1 only if
    something fixable exists. Baseline long-line warnings must not trip it.
    """

    def test_fixable_diagnostic_is_reported_but_file_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.hpp")
            with open(p, "wb") as f:
                f.write(b"if(x) { y; }\n")
            diags = sl.process_file(p, fix=False)
            self.assertTrue(any(d.fixable for d in diags))
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"if(x) { y; }\n")

    def test_long_line_only_file_has_no_fixable(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.hpp")
            with open(p, "w") as f:
                f.write("// " + ("x" * 90) + "\n")
            diags = sl.process_file(p, fix=False)
            self.assertFalse(any(d.fixable for d in diags))

    def test_check_main_passes_on_long_line_only(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.hpp")
            with open(p, "w") as f:
                f.write("// " + ("x" * 90) + "\n")
            script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "style_lint.py"
            )
            r = subprocess.run(
                [sys.executable, script, "--check", td],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)

    def test_check_main_fails_on_fixable(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.hpp")
            with open(p, "wb") as f:
                f.write(b"if(x) { y; }\n")
            script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "style_lint.py"
            )
            r = subprocess.run(
                [sys.executable, script, "--check", td],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 1, msg=r.stdout + r.stderr)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"if(x) { y; }\n")


class TestAtomicWrite(unittest.TestCase):
    def test_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.txt")
            with open(p, "w") as f:
                f.write("old content")
            sl.atomic_write(p, "new content")
            with open(p) as f:
                self.assertEqual(f.read(), "new content")

    def test_no_temp_file_left_behind_on_success(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.txt")
            sl.atomic_write(p, "hello")
            self.assertEqual(os.listdir(td), ["f.txt"])

    def test_preserves_file_mode(self):
        if sys.platform == "win32":
            self.skipTest("POSIX file modes not meaningful on Windows")
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.txt")
            with open(p, "w") as f:
                f.write("old")
            os.chmod(p, 0o640)
            sl.atomic_write(p, "new")
            got = stat.S_IMODE(os.stat(p).st_mode)
            self.assertEqual(got, 0o640)


class TestDigitSeparators(unittest.TestCase):
    """C++14 lets `'` act as a digit separator inside numeric literals.
    The tokenizer must not mistake those for char-literal openers."""

    def test_decimal_separator(self):
        # 1'000 is a pp-number; the apostrophe is code, not a char-lit
        # start, so the trailing semicolon must stay code.
        mask = sl.classify("int x = 1'000;")
        self.assertEqual(mask[-1], "c")
        self.assertNotIn("s", mask)

    def test_hex_separator(self):
        mask = sl.classify("auto m = 0xFF'00'FF;")
        self.assertEqual(mask[-1], "c")
        self.assertNotIn("s", mask)

    def test_odd_number_of_separators_does_not_poison_semicolon(self):
        mask = sl.classify("auto m = 0xFF'00'FF'00;")
        self.assertEqual(mask[-1], "c")

    def test_u8_char_literal_is_not_a_separator(self):
        # u8'a' is a UTF-8 char literal. The `'` must open a char-lit
        # even though the preceding char (`8`) is a digit.
        mask = sl.classify("auto c = u8'a';")
        self.assertIn("s", mask)
        self.assertEqual(mask[-1], "c")

    def test_L_char_literal(self):
        mask = sl.classify("wchar_t c = L'x';")
        self.assertIn("s", mask)

    def test_fix_rewrites_if_paren_after_digit_separator(self):
        text = "int m = 1'000; if(x) { y; }\n"
        new_text, _ = sl.lint_text(text, fix=True)
        self.assertIn("if (", new_text)
        self.assertIn("1'000", new_text)


class TestEdgeCases(unittest.TestCase):
    def test_empty_file(self):
        new_text, diags = sl.lint_text("", fix=True)
        self.assertEqual(new_text, "")
        self.assertEqual(diags, [])

    def test_lone_newline(self):
        new_text, diags = sl.lint_text("\n", fix=True)
        self.assertEqual(new_text, "\n")
        self.assertEqual([d.code for d in diags], [])

    def test_long_line_with_less_than_comparison_is_flagged(self):
        # A single-line `<` comparison must not suppress long-line on
        # its own line. This is the regression test for the template
        # context heuristic that used to trip on any `<`.
        long_cond = "if (a < " + "bbbbbbbb + " * 8 + "0) { }"
        self.assertGreater(len(long_cond), 80)
        _, diags = sl.lint_text(long_cond + "\n", fix=False)
        self.assertTrue(
            any(d.code == "long-line" for d in diags),
            msg=f"long line with `<` was incorrectly suppressed; diags={diags}",
        )

    def test_template_identifier_not_false_positive(self):
        # `myTemplateCount` is an identifier that contains the substring
        # 'template'. The old substring check would skip the brace fix on
        # this line; the word-boundary regex must let it through.
        self.assertFalse(
            sl._is_template_context_line("int myTemplateCount = 0;")
        )
        self.assertFalse(
            sl._is_template_context_line("void templateRegistration();")
        )
        self.assertTrue(
            sl._is_template_context_line("template<typename T> void foo();")
        )

        text = "if (myTemplateCount < 10) {\n  y;\n}\n"
        new_text, diags = sl.lint_text(text, fix=True)
        self.assertIn("if (myTemplateCount < 10)\n{", new_text)
        self.assertTrue(any(d.code == "brace-same-line" for d in diags))


if __name__ == "__main__":
    unittest.main()
