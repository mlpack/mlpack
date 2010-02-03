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
 * @file string.h
 *
 * Simple non-stl string support.
 */

#ifndef COLLECTIONS_STRING_H
#define COLLECTIONS_STRING_H

#include "fastlib/base/base.h"
#include "fastlib/col/arraylist.h"
//#include "arraylist.h"

#include <cstring>

/**
 * Non-stl string with some simple features.
 *
 * The motivation for this is its ability to work well with ArrayList and
 * supports our coding conventions.  Beyond that, it has a few convenient
 * tokenizers that you may find useful.  Finally, it is non-templated,
 * so compiler errors are easier to understand.
 *
 * WARNING: This has not gone through rigorous testing -- we expect it to
 * work, but there may be some issues.  You will be just fine using the
 * STL string if you need string processing.
 */
class String {
 private:
  ArrayList<char> array_;

  OBJECT_TRAVERSAL(String) {
    OT_OBJ(array_);
  }
  
  OT_CUSTOM_PRINT(String) {
    const char *c_str = this->c_str();
    OT_OBJ(c_str);
  }

 public:
  /**
   * Implicit conversion constructor.
   */
  String(const char *s) {
    Copy(s);
  }

  /**
   * Initialize to empty string.
   */
  void Init() {
    array_.Init(1);
    array_[0] = 0;
  }

  /**
   * Allocate a buffer length -- this string won't really be a valid
   * string until you fill the entire buffer.
   */
  void Init(int finalLength) {
    array_.Init(finalLength + 1);
  }
  
  /**
   * Initialize the string in sprintf style.
   *
   * @param format the printf-style format string and arguments
   */
  COMPILER_PRINTF(2, 3)
  const String& InitSprintf(const char *format, ...);
  
  /**
   * Initializes as a copy of an existing region of characters.
   * The existing array does not need to be null terminated.
   */
  void Copy(const char *str_region_begin, index_t len) {
    array_.Init(len + 1);
    mem::Copy(array_.begin(), str_region_begin, len);
    Terminate();
  }
  
  /**
   * Initializes as a copy of a c-style string.
   */
  void Copy(const char *str) {
    array_.InitCopy(str, index_t(strlen(str) + 1));
  }
  
  /**
   * Free up the string so you can reinitialize this to another string.
   */
  void Destruct() {
    array_.Renew();
  }
  
  /**
   * Swaps with another string.
   */
  void Swap(String* other) {
    array_.Swap(&other->array_);
  }
  
  /**
   * Steals the string the other is pointing to, and "uninitializes"
   * the string.
   */
  void StealDestruct(String *other) {
    array_.InitSteal(&other->array_);
    other->Destruct();
  }
  
  /**
   * 
   */
  void Steal(ArrayList<char> *null_terminated_char_list) {
    array_.InitSteal(null_terminated_char_list);
  }
  
  void Steal(char *str, index_t len, index_t capacity) {
    array_.InitSteal(str, len + 1, capacity);
  }
  
  void Steal(char *str, index_t len) {
    array_.InitSteal(str, len + 1, len + 1);
  }
  
  void Steal(char *str) {
    Steal(str, strlen(str));
  }
  
  /** Implicit conversion to c-string. */
  operator const char * () const
   { return array_.begin(); }
  
  /** Returns the internally represented c-string. */
  const char* c_str() const
   { return array_.begin(); }
  /** Returns the internally represented c-string. */
  char* c_str()
   { return array_.begin(); }
  
  /** Returns a pointer to the first character. */
  char* begin()
   { return array_.begin(); }
  /** Returns a pointer to the first character. */
  const char* begin() const   
   { return array_.begin(); }
  /** Returns a pointer to the null terminator character. */
  const char *end() const {
    return array_.end() - 1;
  }
  /** Returns a pointer to the null terminator character. */
  char *end() {
    return array_.end() - 1;
  }
  
  /** Returns the length of the string. */
  index_t length() const
   { return array_.size() - 1; }
  
  /**
   * Reduces the length of the string to a particular
   * location within this string (power user).
   *
   * @param s a pointer to a character in this string
   */
  void Truncate(const char *s) {
    array_.Resize(s - begin() + 1);
    Terminate();
  }
  
  /**
   * Truncates this string at a given length.
   *
   * @param newlen the new length to truncate to
   */
  void Truncate(index_t newlen) {
    array_.Resize(newlen + 1);
    Terminate();
  }
  
  /**
   * Sets the length of the string, if you expect to write past the end
   * of this string (power user).
   */
  void SetLength(index_t newlen) {
    array_.Resize(newlen + 1);
    Terminate();
  }
  
  /**
   * Sets the length to the strlen of the string (power user, when
   * boulding strings).
   */
  void FixLength()
   { array_.Resize(strlen(array_.begin()) + 1); }
  
  /**
   * Puts a null terminator at the end of the string.
   */
  void Terminate() {
    array_.back() = 0;
  }

  /**
   * Minimizes the memory used after a lot of dynamic resizing.
   */
  void Trim()
   { array_.Trim(); }
  
  index_t Find(char c) const {
    return IndexFromPtr(strchr(array_.begin(), c));
  }
  index_t FindR(char c) const {
    return IndexFromPtr(strrchr(array_.begin(), c));
  }
  index_t Find(const char* s) const {
    return IndexFromPtr(strstr(array_.begin(), s));
  }
  index_t FindAny(const char *char_set, index_t skip_initial = 0) const;
  
  index_t IndexFromPtr(const char *position) const {
    if (unlikely(!position)) {
      return -1;
    } else {
      return index_t(position - begin());
    }
  }

  /** Returns true if this string is zero length. */
  bool is_empty() const
   { return array_.size() == 1; }
  
  /**
   * Appends another string to the end of this (power user).
   *
   * Amortized O(strlen(s)).
   * You pass in the length of the string for speed reasons.
   *
   * @param add_str null-terminated string
   * @param add_length the length of the string
   */
  void Append(const char *add_str, index_t add_length) {
    index_t mysize = array_.size();
    array_.Resize(mysize + add_length);
    strcpy(array_.begin() + mysize - 1, add_str);
  }
  
  /**
   * Appends another string to the end of this.
   * Amortized O(strlen(s)).
   */
  void Append(const char *add_str) {
    Append(add_str, strlen(add_str));
  }
  
  /**
   * Appends a character to the end of this string.
   * Amortized O(1).
   */
  void Append(char c) {
    array_.back() = c;
    array_.PushBackCopy('\0');
  }
  
  /**
   * Appends another string to the end of this.
   * Amortized O(o.length()).
   */
  void Append(const String& str) {
    Append(str.array_.begin(), str.length());
  }
  
  
  /**
   * Appends another string to the end of this.
   * Amortized O(o.length()).
   */
  const String& operator += (const String& o) {
    Append(o);
    return *this;
  }
  
  /**
   * Appends a character to the end of this string.
   * Amortized O(1).
   */
  const String& operator += (char c) {
    Append(c);
    return *this;
  }
  
  /**
   * Appends another string to the end of this.
   * Amortized O(strlen(s)).
   */
  const String& operator += (const char *s) {
    Append(s);
    return *this;
  }
  
  /**
   * Gets individual characters.
   */
  char operator [] (index_t index) const
   { return array_[index]; }
  /**
   * Gets individual characters, and allows modification.
   */
  char& operator [] (index_t index)
   { return array_[index]; }
  
  /**
   * Splits up the string like strtok.
   *
   * Starts at begin_index, skips initial delimeters.
   * Proceeds to find any character in the delimeters or donechars.
   * If it's a delimeter and no more max_portions has not been exceeded,
   * a token is made.  Processing stops when donechars are found.
   *
   * Example:
   *
   * @code
   * String s;
   * ArrayList<String> list;
   * s.Copy("XXX,,,a, b;c; d! qrstuvk")
   * list.Init();
   * s.Split(3, ",; ", "!", 3, &list);
   * // The result is: "a", "b", "c; d". (notice last two are one string)
   * @endcode
   *
   * @param start_index the index to start at (initial characters to skip)
   * @param delimeters the delimeter characters you are interested in
   * @param donechars characters on which to stop (you can use "")
   * @param max_portions the maximum number of portions (0 means unlimited)
   * @param result the ArrayList to append results to (must be initialized)
   * @return the index at which splitting stopped
   */
  index_t Split(index_t start_index, const char *delimeters,
      const char *donechars, index_t max_portions,
      ArrayList<String> *result) const;
  
  /**
   * Splits up the string like strtok.
   *
   * Example:
   *
   * @code
   * String s;
   * ArrayList<String> list;
   * s.Copy(",,a, b;c, d,,")
   * list.Init();
   * s.Split(",; ", &list);
   * // The result is: "a", "b", "c", "d".
   * @endcode
   *   *
   * @param delimeters the delimeter characters you are interested in
   * @param result the ArrayList to append results to (must be initialized)
   * @return the index at which splitting stopped
   */
  index_t Split(const char *delimeters, ArrayList<String> *result) const {
    return Split(0, delimeters, "", 0, result);
  }
  
  /**
   * Creates a new string with the specified characters removed from the left
   * part of the string.
   */
  void TrimLeft(const char *delimeters, String *result) const;

  /**
   * Creates a new string with the specified characters removed from the right
   * part of the string.
   */
  void TrimRight(const char *delimeters, String *result) const;

  /**
   * Creates a new string with the specified characters removed from the left
   * and right parts of the string.
   */
  void Trim(const char *delimeters, String *result) const;

  /**
   * Compares two strings case-insensitively.
   */
  int CompareNoCase(const char *s) const {
    return strcasecmp(array_.begin(), s);
  }
  /**
   * Checks if two strings are equal.
   */
  bool EqualsNoCase(const char *s) const {
    return CompareNoCase(s) == 0;
  }
  
  /**
   * Checks if this string begins with another string.
   */
  bool StartsWith(const char *s) const {
    return strncmp(array_.begin(), s, strlen(s)) == 0;
  }
  
  /**
   * Compares two strings.
   *
   * If I am the lesser string, I return negative.
   */
  int CompareTo(const String& other) const
   { return strcmp(begin(), other.begin()); }
  /**
   * Compares two strings.
   *
   * If I am the lesser string, I return negative.
   */
  int CompareTo(const char* s) const
   { return strcmp(begin(), s); }

  friend bool operator < (const String& a, const String& b) {
     return strcmp(a.begin(), b.begin()) < 0;
  }
  EXPAND_LESS_THAN(String);
  friend bool operator == (const String& a, const String& b) {
     return strcmp(a.begin(), b.begin()) == 0;
  }
  EXPAND_EQUALS(String);
  
  friend bool operator < (const char *a, const String& b) {
     // >
     return strcmp(a, b.begin()) < 0;
  }
  EXPAND_HETERO_LESS_THAN(const char *, String);
  friend bool operator < (const String& a, const char *b) {
     // >
     return strcmp(a.begin(), b) < 0;
  }
  EXPAND_HETERO_LESS_THAN(String, const char *);
  friend bool operator == (const String& a, const char *b) {
     return strcmp(a.begin(), b) == 0;
  }
  EXPAND_HETERO_EQUALS(String, const char *);
  
  friend bool operator < (char a, const String& b) {
     // >
     DEBUG_ASSERT(b != '\0');
     return a < b[0] || (unlikely(a == b[0]) && b[1] != '\0');
  }
  EXPAND_HETERO_LESS_THAN(char, String);
  friend bool operator < (const String& a, char b) {
     // >
     DEBUG_ASSERT(b != '\0');
     return a[0] < b;
  }
  EXPAND_HETERO_LESS_THAN(String, char);
  friend bool operator == (const String& a, char b) {
     DEBUG_ASSERT(b != '\0');
     return a[0] == b && a[1] == '\0';
  }
  EXPAND_HETERO_EQUALS(String, char);
};

#endif
