/**
 * @file textfile.hpp
 *
 * Small wrappers for text files.
 * The most useful thing here is the ReadLine function.
 *
 * @bug These routines fail when trying to read files linewise that use the Mac
 * eol '\r'.  Both Windows and Unix eol ("\r\n" and '\n') work.  Use the
 * programs 'dos2unix' or 'tr' to convert the '\r's to '\n's.
 *
 */

#ifndef __MLPACK_CORE_FILE_TEXTFILE_HPP
#define __MLPACK_CORE_FILE_TEXTFILE_HPP

#include "../io/cli.hpp"
#include "../io/log.hpp"

#include <cstdio>
#include <ctype.h>
#include <stdarg.h>

#include <string>
#include <vector>

namespace mlpack {

/**
 * Helper for reading text files.
 *
 * Files are closed automatically when they fall out of scope, though
 * you may choose to close it yourself at no harm.
 */
class TextLineReader {

 private:
  FILE *f_;
  std::string line_;
  int line_num_;
  bool has_line_;
  std::string fname_;

 public:
  /** Creates an unitialized object. */
  TextLineReader() {
    f_ = NULL;
  }

  /**
   * Automatically closes the file.
   */
  ~TextLineReader() {
    if (f_)
      ::fclose(f_);
  }

  /**
   * Opens a file.
   *
   * @return success value
   */
  bool Open(const char *fname);

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
   * Return the name of the file we are working with.
   * This will return NULL if no file has been opened yet.
   */
  const std::string& filename() const {
    return fname_;
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
  std::string& Peek() {
    return line_;
  }

  /**
   * Gets the current line.
   */
  const std::string& Peek() const {
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
  __attribute__((format(printf, 2, 3))) void Error(const char *msg, ...);

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
  std::string next_;
  TokenType next_type_;
  std::string cur_;
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
    if (f_ != NULL) {
      (void) fclose(f_);
    }
    f_ = NULL;
  }

  bool Open(const char *fname,
      const char *comment_chars = "", const char *ident_extra = "",
      int features = 0);


  const std::string& Peek() const {
    return next_;
  }

  TokenType PeekType() const {
    return next_type_;
  }

  const std::string& Current() const {
    return cur_;
  }

  TokenType CurrentType() const {
    return cur_type_;
  }

  void Gobble();

  bool MoreTokens() const {
    return next_type_ != END;
  }

  bool Match(const std::string exact) {
    if (next_ == exact) {
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

  __attribute__((format(printf, 2, 3))) void Error(const char *msg, ...);

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

  char Skip_(std::vector<char>& token);

  char NextChar_(std::vector<char>& token);

  char NextChar_();

  void UndoNextChar_(std::vector<char>& token);

  void Error_(const char *msg, const std::vector<char>& token);

  bool isident_begin_(int c) const {
    return isalpha(c) || (c == '_');
  }

  bool isident_rest_(int c) const {
    return isalnum(c) || (c == '_') || (c != 0 && strchr(ident_extra_, c));
  }

  void ScanNumber_(char c, std::vector<char>& token);

  void ScanString_(char ending, std::vector<char>& token);

  void Scan_(std::vector<char>& token);
};

/**
 * Helper for writing text fo a file.
 */
class TextWriter {

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
      mlpack::Log::Assert(fclose(f_) >= 0);
      mlpack::Log::Assert(fclose(f_) >= 0, "File close failed!");
    }
    f_ = NULL;
  }

  /**
   * Opens a file by name (initializer).
   *
   * @return success or failure
   */
  bool Open(const char *fname) {
    f_ = ::fopen(fname, "w");
    return (!f_) ? false : true;
  }

  /**
   * Explicitly closes the file.
   */
  bool Close() {
    int rv = fclose(f_);
    f_ = NULL;
    return (rv < 0) ? false : true;
  }

  bool Printf(const char *format, ...);

  bool Write(const char *s) {
    return (fputs(s, f_) > 0);
  }

  bool Write(int i) {
    return (fprintf(f_, "%d", i) > 0);
  }

  bool Write(unsigned int i) {
    return (fprintf(f_, "%u", i) > 0);
  }

  bool Write(long i) {
    return (fprintf(f_, "%ld", i) > 0);
  }

  bool Write(unsigned long i) {
    return (fprintf(f_, "%lu", i) > 0);
  }

  bool Write(long long i) {
    return (fprintf(f_, "%lld", i) > 0);
  }

  bool Write(unsigned long long i) {
    return (fprintf(f_, "%llu", i) > 0);
  }

  bool Write(double d) {
    return (fprintf(f_, "%.15e", d) > 0);
  }
};

}; // namespace mlpack

#endif // __MLPACK_CORE_FILE_TEXTFILE_HPP
