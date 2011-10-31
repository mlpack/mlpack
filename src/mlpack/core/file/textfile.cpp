/**
 * @file textfile.cpp
 *
 * Implementations for the text-based file I/O helper classes.
 *
 * @bug These routines fail when trying to read files linewise that use the Mac
 * eol '\r'.  Both Windows and Unix eol ("\r\n" and '\n') work.  Use the
 * programs 'dos2unix' or 'tr' to convert the '\r's to '\n's.
 *
 */

#include "textfile.hpp"

#include "../io/cli.hpp"
#include "../io/log.hpp"
#include <ctype.h>
#include <iostream>

using namespace mlpack;

/*
char *TextTokenizer::ReadLine() {
  char *buf = NULL;
  size_t size = 0;
  size_t len = 0;
  const size_t extra = 64;
  int c;

  for (;;) {
    c = getc(f_);

    if (c == '\r') {
      c = getc(f_);
      if (c != '\n') {
        ungetc(c, f_);
      }
      break;
    } else if ((c == '\n') {
      break;
    } else if (c == EOF) {
      if (len == 0) {
        return NULL;
      } else {
        break;
      }
    }

    len++;

    if (size <= len) {
      size = len * 2 + extra;
      buf = mem::Realloc(buf, size);
    }

    buf[len-1] = c;
  }

  if (len == 0) {
    // special case: empty line
    buf = mem::Alloc<char>(1);
  }

  buf[len] = '\0';

  return buf;
}
*/

void TextLineReader::Error(const char *format, ...) {
  va_list vl;

  // TODO: Use a warning propagation system
  fprintf(stderr, ".| %d: %s\nX|  `-> ", line_num_, line_.c_str());

  va_start(vl, format);
  vfprintf(stderr, format, vl);
  va_end(vl);

  fprintf(stderr, "\n");
}

bool TextLineReader::Open(const char *fname) {
  f_ = fopen(fname, "r");
  fname_ = fname;
  line_num_ = 0;
  has_line_ = false;

  if (f_ == NULL) {
    return false;
  } else {
    Gobble();
    return true;
  }
}

bool TextLineReader::Gobble() {
  char *ptr = ReadLine_();

  if (ptr != NULL) {
    line_ = ptr;
    has_line_ = true;
    line_num_++;
    delete[] ptr;
    return true;
  } else {
    has_line_ = false;
    delete[] ptr;
    return false;
  }
}

char *TextLineReader::ReadLine_() {
  char *buf = NULL;
  size_t size = 1;
  size_t len = 0;
#ifdef DEBUG
  const size_t extra = 10;
#else
  const size_t extra = 80;
#endif

  for (;;) {
    size = size * 2 + extra;

    // Reallocate
    char* newbuf = new char[size];
    if (buf != NULL) {
      memcpy(newbuf, buf, len * sizeof(char));
      delete[] buf;
    }
    buf = newbuf;

    //! doesn't handle mac eol - OK?
    char *result = ::fgets(buf + len, size - len, f_);
    if (len == 0 && result == NULL) {
      delete[] buf;
      return NULL;
    }
    len += strlen(buf + len);
    if (buf[len - 1] == '\r') { // is there a following \n we didn't pick up?
      // we'll peek at the next character and keep it if it's \n, but move the
      // pointer back a position if it isn't
      char tmp = fgetc(f_);
      if(tmp == '\n') { // append to end
        size++;

        char* newbuf = new char[size * sizeof(char)];
        memcpy(newbuf, buf, (size - 1) * sizeof(char));
        delete[] buf;
        buf = newbuf;
        buf[len] = tmp;
      } else {
        // go back a character
        fseek(f_, -1, SEEK_CUR);
      }
    }

    if (len < size - 1 || buf[len - 1] == '\r' || buf[len - 1] == '\n') {
      while (len && (buf[len-1] == '\r' || buf[len-1] == '\n')) {
        len--;
      }
      buf[len] = '\0';
      return buf;
    }
  }
}

bool TextTokenizer::Open(const char *fname,
    const char *comment_chars_in, const char *ident_extra_in,
    int features_in) {
  next_ = "";
  cur_ = "";
  next_type_ = END;
  cur_type_ = END;
  comment_start_ = comment_chars_in;
  features_ = features_in;
  ident_extra_ = ident_extra_in;
  line_ = 1;

  f_ = fopen(fname, "r");

  if (f_ == NULL) {
    return false;
  } else {
    Gobble();
    return true;
  }
}

char TextTokenizer::NextChar_() {
  int c = GetChar_();

  if (c != EOF && (strchr(comment_start_, c) != NULL)) {
    do {
      c = GetChar_();
    } while (c != EOF && c != '\r' && c != '\n');
  }

  if (c == EOF) {
    c = 0;
  }

  return c;
}

char TextTokenizer::NextChar_(std::vector<char>& token) {
  char c = NextChar_();

  token.push_back(c);

  return c;
}

char TextTokenizer::Skip_(std::vector<char>& token) {
  int c;

  while (1) {
    c = NextChar_();
    if (!isspace(c)) {
      break;
    }

    if (c == '\r' || c == '\n') {
      if (c == '\r') {
        c = NextChar_();
        if (c != '\n') {
          Unget_(c);
        }
      }
      line_++;
      if ((features_ & WANT_NEWLINE)) {
        c = '\n';
        break;
      }
    }
  }

  token.push_back(char(c));

  return char(c);
}

void TextTokenizer::UndoNextChar_(std::vector<char>& token) {
  char c;
  c = token.back();
  token.pop_back();
  if (c != 0) { /* don't put EOF back on the stream */
    Unget_(c);
  }
}

void Sanitize(const std::string& src, std::string& dest) {
  for (size_t i = 0; i < src.length(); i++) {
    char c = src[i];

    if (isgraph(c) || c == ' ' || c == '\t') {
      dest += c;
    } else if (isspace(c)) {
      dest += "<whitespace>";
    } else {
      dest += "<nonprint>";
    }
  }
}

void TextTokenizer::Error(const char *format, ...) {
  va_list vl;
  std::string cur_sanitized;
  std::string next_sanitized;

  Sanitize(cur_, cur_sanitized);
  Sanitize(next_, next_sanitized);

  // TODO: Use a warning propagation system
  fprintf(stderr, ".| %d: %s <-HERE-> %s\nX|  `-> ", line_,
      cur_sanitized.c_str(), next_sanitized.c_str());

  va_start(vl, format);
  vfprintf(stderr, format, vl);
  va_end(vl);

  fprintf(stderr, "\n");
}

void TextTokenizer::Error_(const char *msg, const std::vector<char>& token) {
  next_type_ = INVALID;

  printf("size is %zu, token[0] = %d\n", token.size(), token[0]);
  Error("%s", msg);
  next_.clear();
}

void TextTokenizer::ScanNumber_(char c, std::vector<char>& token) {
  bool dot = false;
  bool floating = false;

  while (1) {
    if (c == '.') {
      /* handle a period */
      if (dot) {
        Error_("Multiple decimal points in a float", token);
        return;
      }
      dot = true;
      floating = true;
    } else if (isdigit(c)) {
      /* keep on processing digits */
    } else if (c == 'e' || c == 'E') {
      /* exponent - read exponent and finish */
      c = NextChar_(token);
      if (c == '+' || c == '-') {
        c = NextChar_(token);
      }
      while (isdigit(c)) {
        c = NextChar_(token);
      }
      floating = true;
      break;
    } else {
      /* non numeric */
      break;
    }

    c = NextChar_(token);
  }

  if (c == 'f' || c == 'F') {
    // It's labelled a float.  Gobble and go.
    floating = true;
  } else if (isspace(c) || ispunct(c)) {
    UndoNextChar_(token);
  } else {
    Error_("Invalid character while parsing number", token);
  }

  if (floating) {
    next_type_ = DOUBLE;
  } else {
    next_type_ = INTEGER;
  }
}

void TextTokenizer::ScanString_(char ending, std::vector<char>& token) {
  int c;

  while (1) {
    c = NextChar_(token);

    if (c == 0) {
      Error_("Unterminated String", token);
      UndoNextChar_(token);
      return;
    }

    if (c == ending) {
      next_type_ = STRING;
      return;
    }
  }
}

void TextTokenizer::Scan_(std::vector<char>& token) {
  char c = Skip_(token);

  if (c == 0) {
    // Makes token's capacity = 0
    token.clear();
    next_type_ = END;
    return;
  } else if (c == '.' || isdigit(c)) {
    ScanNumber_(c, token);
  } else if (isident_begin_(c)) {
    while (isident_rest_(NextChar_(token))) {}
    UndoNextChar_(token);
    next_type_ = IDENTIFIER;
  } else if (ispunct(c) || isspace(c)) {
    if (c == '"' || c == '\'') {
      ScanString_(c, token);
    } else if (c == '+' || c == '-') {
      c = NextChar_(token);
      if (c == '.' || isdigit(c)) {
        ScanNumber_(c, token);
      } else {
        UndoNextChar_(token);
      }
    } else {
      next_type_ = PUNCT;
    }
  } else {
    Error_("Unknown Character", token);
  }
}

void TextTokenizer::Gobble() {

  cur_.assign(next_);
  next_.clear();
  cur_type_ = next_type_;

  std::vector<char> token;
  Scan_(token);
  token.push_back('\0');

  next_ = &token.front();

  mlpack::Log::Assert(next_.length() == strlen(next_.c_str()));
}

bool TextWriter::Printf(const char *format, ...) {
  int rv;

  va_list vl;

  va_start(vl, format);
  rv = vfprintf(f_, format, vl);
  va_end(vl);

  return (rv > 0);
}

