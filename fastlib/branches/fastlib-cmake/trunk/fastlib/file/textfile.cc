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
 * @file textfile.cc
 *
 * Implementations for the text-based file I/O helper classes.
 * 
 * @bug These routines fail when trying to read files linewise that use the Mac
 * eol '\r'.  Both Windows and Unix eol ("\r\n" and '\n') work.  Use the
 * programs 'dos2unix' or 'tr' to convert the '\r's to '\n's.
 *
 */

#include "fastlib/file/textfile.h"
//#include "textfile.h"

#include <ctype.h>

/*
char *TextTokenizer::ReadLine() {
  char *buf = NULL;
  size_t size = 0;
  size_t len = 0;
  const size_t extra = 64;
  int c;
  
  for (;;) {
    c = getc(f_);
    
    if (unlikely(c == '\r')) {
      c = getc(f_);
      if (c != '\n') {
        ungetc(c, f_);
      }
      break;
    } else if (unlikely(c == '\n')) {
      break;
    } else if (unlikely(c == EOF)) {
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

success_t TextLineReader::Open(const char *fname) {
  f_ = fopen(fname, "r");
  line_num_ = 0;
  has_line_ = false;
  line_.Init();
  
  if (unlikely(f_ == NULL)) {
    return SUCCESS_FAIL;
  } else {
    Gobble();
    return SUCCESS_PASS;
  }
}  

bool TextLineReader::Gobble() {
  char *ptr = ReadLine_();
  
  line_.Destruct();
  
  if (likely(ptr != NULL)) {
    line_.Steal(ptr);
    has_line_ = true;
    line_num_++;
    return true;
  } else {
    line_.Init();
    has_line_ = false;
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
    buf = mem::Realloc(buf, size);
    //! doesn't handle mac eol - OK?
    char *result = ::fgets(buf + len, size - len, f_); 
    if (len == 0 && result == NULL) {
      mem::Free(buf);
      return NULL;
    }
    len += strlen(buf + len);
    
    if (len < size - 1 || buf[len - 1] == '\r' || buf[len - 1] == '\n') {
      while (len && (buf[len-1] == '\r' || buf[len-1] == '\n')) {
        len--;
      }
      buf[len] = '\0';
      return buf;
    }
  }
}

success_t TextTokenizer::Open(const char *fname,
    const char *comment_chars_in, const char *ident_extra_in,
    int features_in) {
  next_.Copy("");
  cur_.Copy("");
  next_type_ = END;
  cur_type_ = END;
  comment_start_ = comment_chars_in;
  features_ = features_in;
  ident_extra_ = ident_extra_in;
  line_ = 1;
  
  f_ = fopen(fname, "r");
  
  if (unlikely(f_ == NULL)) {
    return SUCCESS_FAIL;
  } else {
    Gobble();
    return SUCCESS_PASS;
  }
}

char TextTokenizer::NextChar_() {
  int c = GetChar_();
  
  if (c != EOF && unlikely(strchr(comment_start_, c) != NULL)) {
    do {
      c = GetChar_();
    } while (likely(c != EOF) && likely(c != '\r') && likely(c != '\n'));
  }
  
  if (unlikely(c == EOF)) {
    c = 0;
  }
  
  return c;
}

char TextTokenizer::NextChar_(ArrayList<char> *token) {
  char c = NextChar_();

  token->PushBackCopy(c);
  
  return c;
}

char TextTokenizer::Skip_(ArrayList<char> *token) {
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
  
  token->PushBackCopy(char(c));
  
  return char(c);
}

void TextTokenizer::UndoNextChar_(ArrayList<char> *token) {
  char c;
  token->PopBackInit(&c);
  if (c != 0) { /* don't put EOF back on the stream */
    Unget_(c);
  }
}

void Sanitize(const String& src, String* dest) {
  dest->Init();
  
  for (index_t i = 0; i < src.length(); i++) {
    char c = src[i];
    
    if (isgraph(c) || c == ' ' || c == '\t') {
      *dest += c;
    } else if (isspace(c)) {
      *dest += "<whitespace>";
    } else {
      *dest += "<nonprint>";
    }
  }
}

void TextTokenizer::Error(const char *format, ...) {
  va_list vl;
  String cur_sanitized;
  String next_sanitized;
  
  Sanitize(cur_, &cur_sanitized);
  Sanitize(next_, &next_sanitized);
  
  // TODO: Use a warning propagation system
  fprintf(stderr, ".| %d: %s <-HERE-> %s\nX|  `-> ", line_,
      cur_sanitized.c_str(), next_sanitized.c_str());
  
  va_start(vl, format);
  vfprintf(stderr, format, vl);
  va_end(vl);
  
  fprintf(stderr, "\n");
}

void TextTokenizer::Error_(const char *msg, const ArrayList<char>& token) {
  next_type_ = INVALID;
  
  printf("size is %"LI"d, token[0] = %d\n", token.size(), token[0]);
  next_.Copy(token.begin(), token.size());
  Error("%s", msg);
  next_.Destruct();
}

void TextTokenizer::ScanNumber_(char c, ArrayList<char> *token) {
  bool dot = false;
  bool floating = false;
  
  while (1) {
    if (unlikely(c == '.')) {
      /* handle a period */
      if (unlikely(dot)) {
        Error_("Multiple decimal points in a float", *token);
        return;
      }
      dot = true;
      floating = true;
    } else if (likely(isdigit(c))) {
      /* keep on processing digits */
    } else if (unlikely(c == 'e' || c == 'E')) {
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
    Error_("Invalid character while parsing number", *token);
  }
  
  if (floating) {
    next_type_ = DOUBLE;
  } else {
    next_type_ = INTEGER;
  }
}

void TextTokenizer::ScanString_(char ending, ArrayList<char> *token) {
  int c;
  
  while (1) {
    c = NextChar_(token);
    
    if (c == 0) {
      Error_("Unterminated String", *token);
      UndoNextChar_(token);
      return;
    }
    
    if (c == ending) {
      next_type_ = STRING;
      return;
    }
  }
}

void TextTokenizer::Scan_(ArrayList<char> *token) {
  char c = Skip_(token);
  
  if (c == 0) {
    token->Clear();
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
    Error_("Unknown Character", *token);
  }
}

void TextTokenizer::Gobble() {
  cur_.Destruct();
  cur_.StealDestruct(&next_);
  cur_type_ = next_type_;
  
  ArrayList<char> token;
  token.Init();
  Scan_(&token);
  token.PushBackCopy('\0');
  next_.Steal(&token);
  DEBUG_ASSERT(next_.length() == index_t(strlen(next_.c_str())));
}

success_t TextWriter::Printf(const char *format, ...) {
  int rv;
  
  va_list vl;
  
  va_start(vl, format);
  rv = vfprintf(f_, format, vl);
  va_end(vl);
  
  return SUCCESS_FROM_C(rv);
}

