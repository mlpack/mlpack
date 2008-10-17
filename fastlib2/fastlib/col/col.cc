/**
 * @file col.cc
 *
 * Non-templated implementations of various collections.
 */

#include "arraylist.h"
#include "col_string.h"

#include <stdarg.h>

const String& String::InitSprintf(const char *format, ...) {
  int size = 128;
  int len;
  char *s = mem::Alloc<char>(size);
  va_list vl;
  
  while (1) {
    va_start(vl, format);
    len = vsnprintf(s, size, format, vl);
    va_end(vl);
    
    if (likely(len > -1) && likely(len < size)) {
      break;
    }
    
    if (len > -1) {
      // snprintf is telling us exactly how long it should be
      size = len + 1;
    } else {
      // snprintf gave up and said it's too small, let's try again
      size *= 2;
    }
    
    s = mem::Realloc(s, size);
  }
  
  Steal(s, len, size);
  
  return *this;
}

index_t String::FindAny(const char *char_set, index_t skip_initial) const {
  const char *pos = begin() + skip_initial;
  
  DEBUG_BOUNDS(skip_initial, length() + 1);
  
  while (*pos != '\0' && strchr(char_set, *pos) == NULL) {
    pos++;
  }
  
  if (*pos == '\0') {
    return -1;
  } else {
    return pos - begin();
  }
}

index_t String::Split(index_t start_index,
    const char *delimeters,
    const char *donechars,
    index_t max_portions,
    ArrayList<String> *result) const {
  const char *pos = begin() + start_index;
  bool done = false;
  
  DEBUG_BOUNDS(start_index, length() + 1);
  
  do {
    const char *startpos;
    const char *endpos;

    while (*pos != '\0' && strchr(delimeters, *pos) != NULL) {
      pos++;
    }
    
    startpos = endpos = pos;
    
    while (1) {
      if (unlikely(*endpos == '\0') || strchr(donechars, *endpos) != NULL) {
        // strip extra delimeters from right side
        while (endpos > startpos && strchr(delimeters, endpos[-1]) != NULL) {
          endpos--;
        }
        done = true;
        break;
      }
      if (max_portions != 1 && strchr(delimeters, *endpos) != NULL) {
        break;
      }
      endpos++;
    }
    
    pos = endpos;
    
    max_portions--;
    
    if (likely(startpos != endpos)) {
      result->PushBack();
      result->back().Copy(startpos, endpos - startpos);
    }
  } while (!done);
  
  return pos - begin();
}

void String::TrimLeft(const char *delimeters, String *result) const {
  const char *s = begin();
  while (*s != '\0' && strchr(delimeters, *s)) {
    s++;
  }
  result->Copy(s, end() - s);
}

void String::TrimRight(const char *delimeters, String *result) const {
  const char *s = end() - 1;
  const char *b = begin();
  while (s >= b && strchr(delimeters, *s)) {
    s--;
  }
  result->Copy(b, s - b + 1);
}

void String::Trim(const char *delimeters, String *result) const {
  const char *b = begin();
  const char *e = end() - 1;
  while (e >= b && strchr(delimeters, *e)) {
    e--;
  }
  while (e >= b && strchr(delimeters, *b)) {
    b++;
  }
  result->Copy(b, e - b + 1);
}
