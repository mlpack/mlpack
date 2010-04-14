/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file textfile.h
 *
 * Small wrappers for text files.
 * The most useful thing here is the ReadLine function.
 * 
 * @bug These routines fail when trying to read files linewise that use the Mac
 * eol '\r'.  Both Windows and Unix eol ("\r\n" and '\n') work.  Use the
 * programs 'dos2unix' or 'tr' to convert the '\r's to '\n's.
 *
 */

#ifndef FILE_TEXTFILE_H
#define FILE_TEXTFILE_H

#include "../base/base.h"
#include "../col/col_string.h"

#include <cstdio>
#include <ctype.h>
#include <stdarg.h>

/**
 * Helper for reading text files.
 *
 * Files are closed automatically when they fall out of scope, though
 * you may choose to close it yourself at no harm.
 */
class TextLineReader {
  FORBID_ACCIDENTAL_COPIES(TextLineReader);
  
 private:
  FILE *f_;
  String line_;
  int line_num_;
  bool has_line_;
  
 public:
  /** Creates an unitialized object. */
  TextLineReader() {
    f_ = NULL;
  }
  
  /**
   * Automatically closes the file.
   */
  ~TextLineReader() {
    if (f_) {
      ::fclose(f_);
    }
  }
  
  /**
   * Opens a file.
   *
   * @return success value
   */
  success_t Open(const char *fname);
  
  /**
   * Closes the file.
   *
   * (No need for a return value since you are only reading the file.)
   */
  void Close() {
    (void)fclose(f_);
    f_ = NULL;
  }
  
  /**
   * Are there more lines left?
   */
  bool MoreLines() {
    return has_line_;
  }

  /**
   * Returns the currnet line number.
   */
  int line_num() const {
    return line_num_;
  }
  
  /**
   * Gets the current line.
   */
  String& Peek() {
    return line_;
  }
  
  /**
   * Gets the current line.
   */
  const String& Peek() const {
    return line_;
  }
  
  /**
   * Tries to read one line from a file.
   *
   * @return true if a line was returned, false if end of file
   */
  bool Gobble();

  /**
   * Prints a formatted error message with line number info.
   */
  COMPILER_PRINTF(2, 3)
  void Error(const char *msg, ...);
  
 private:
  char *ReadLine_();
};

/**
 * Simple text tokenizer.
 *
 * This tokenizes the input stream.  It will identify the general type of
 * token, and ignore whitespace and different kinds of comments.
 *
 * This has the concept of 'next' token which you can peek at, and the
 * current token.  The current token is always initialized to empty at
 * the very beginning, because you are encouraged to peek ahead one token.
 *
 * Example:
 *
 * @code
 * TextTokenizer tokenizer;
 * tokenizer.Open("file.txt");
 * 
 * if (tokenizer->Match("count")) {
 *   if (tokenizer->Match(TextTokenizer::INTEGER)) {
 *     printf("Found number: %d\n", atoi(tokenizer->Current()));
 *   } else { Error(); }
 * } else { Error(); }
 * @endcode
 */
class TextTokenizer {
  FORBID_ACCIDENTAL_COPIES(TextTokenizer);
 public:
  enum TokenType {
    INVALID = -1,
    END,
    PUNCT,
    IDENTIFIER,
    STRING,
    DOUBLE,
    INTEGER
  };
  
  enum Features {
    WANT_NEWLINE = 0x01
  };
 
 private:
  FILE *f_;
  String next_;
  TokenType next_type_;
  String cur_;
  TokenType cur_type_;
  const char *comment_start_;
  const char *ident_extra_;
  int features_;
  int line_;
  
 public:
  TextTokenizer() {
    f_ = NULL;
  }
  ~TextTokenizer() {
    if (unlikely(f_ != NULL)) {
      (void) fclose(f_);
    }
    DEBUG_POISON_PTR(f_);
  }
  
  success_t Open(const char *fname,
      const char *comment_chars = "", const char *ident_extra = "",
      int features = 0);

  
  const String& Peek() const {
    return next_;
  }
  
  TokenType PeekType() const {
    return next_type_;
  }
  
  const String& Current() const {
    return cur_;
  }
  
  TokenType CurrentType() const {
    return cur_type_;
  }
  
  void Gobble();
  
  bool MoreTokens() const {
    return next_type_ != END;
  }
  
  bool Match(const char *exact) {
    if (next_ == exact) {
      Gobble();
      return true;
    } else {
      return false;
    }
  }
  
  bool MatchNoCase(const char *str) {
    if (next_.EqualsNoCase(str)) {
      Gobble();
      return true;
    } else {
      return false;
    }
  }
  
  bool MatchInteger() {
    return MatchType(INTEGER);
  }
  
  bool MatchDouble() {
    return MatchType(DOUBLE);
  }
  
  bool MatchNumber() {
    return MatchInteger() || MatchDouble();
  }
  
  bool MatchIdentifier() {
    return MatchType(IDENTIFIER);
  }
  
  bool MatchQuasiIdentifier() {
    return MatchIdentifier() || MatchNumber();
  }
  
  bool MatchString() {
    return MatchType(STRING);
  }
  
  bool MatchPunct() {
    return MatchType(PUNCT);
  }
  
  bool MatchType(TokenType type) {
    if (next_type_ == type) {
      Gobble();
      return true;
    } else {
      return false;
    }
  }
  
  int line() const {
    return line_;
  }

  COMPILER_PRINTF(2, 3)
  void Error(const char *msg, ...);
  
 private:
  int GetChar_() {
    return ::getc(f_);
  }
  
  void Unget_(int c) {
    ::ungetc(c, f_);
  }
  
  bool IsEOF_() {
    return ::feof(f_);
  }
  
  char Skip_(ArrayList<char> *token);
  
  char NextChar_(ArrayList<char> *token);

  char NextChar_();
  
  void UndoNextChar_(ArrayList<char> *token);
  
  void Error_(const char *msg, const ArrayList<char>& token);
  
  bool isident_begin_(int c) const {
    return isalpha(c) || unlikely(c == '_');
  }
  
  bool isident_rest_(int c) const {
    return isalnum(c) || unlikely(c == '_') || (c != 0 && strchr(ident_extra_, c));
  }

  void ScanNumber_(char c, ArrayList<char> *token);

  void ScanString_(char ending, ArrayList<char> *token);
  
  void Scan_(ArrayList<char> *token);
};

/**
 * Helper for writing text fo a file.
 */
class TextWriter {
  FORBID_ACCIDENTAL_COPIES(TextWriter);
  
 private:
  FILE *f_;
  
 public:
  /**
   * Creates an uninitialized text writer (you must initialize it).
   */
  TextWriter() {
    f_ = NULL;
  }
  
  /**
   * Automatically closes the file when it gets out of scope; for
   * best error handling, you should call Close first.
   *
   * If you do not explicitly close the file beforehand, this will abort the
   * program on a write error.
   */
  ~TextWriter() {
    if (f_) {
      MUST_PASS(SUCCESS_FROM_C(::fclose(f_)));
    }
    DEBUG_POISON_PTR(f_);
  }
  
  /**
   * Opens a file by name (initializer).
   *
   * @return success or failure
   */
  success_t Open(const char *fname) {
    f_ = ::fopen(fname, "w");
    return (unlikely(!f_)) ? SUCCESS_FAIL : SUCCESS_PASS;
  }
  
  /**
   * Explicitly closes the file.
   */
  success_t Close() {
    int rv = fclose(f_);
    f_ = NULL;
    return unlikely(rv < 0) ? SUCCESS_FAIL : SUCCESS_PASS;
  }
  
  success_t Printf(const char *format, ...);
  
  success_t Write(const char *s) {
    return SUCCESS_FROM_C(fputs(s, f_));
  }
  
  success_t Write(int i) {
    return SUCCESS_FROM_C(fprintf(f_, "%d", i));
  }
  
  success_t Write(unsigned int i) {
    return SUCCESS_FROM_C(fprintf(f_, "%u", i));
  }

  success_t Write(long i) {
    return SUCCESS_FROM_C(fprintf(f_, "%ld", i));
  }
  
  success_t Write(unsigned long i) {
    return SUCCESS_FROM_C(fprintf(f_, "%lu", i));
  }
  
  success_t Write(long long i) {
    return SUCCESS_FROM_C(fprintf(f_, "%lld", i));
  }
  
  success_t Write(unsigned long long i) {
    return SUCCESS_FROM_C(fprintf(f_, "%llu", i));
  }
  
  success_t Write(double d) {
    return SUCCESS_FROM_C(fprintf(f_, "%.15e", d));
  }
};

#endif
