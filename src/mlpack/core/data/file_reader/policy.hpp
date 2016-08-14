// Copyright: (2012-2015) Ben Strasser <code@ben-strasser.net>
// License: BSD-3
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//2. Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
//3. Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef MLPACK_CORE_DATA_FILE_READER_POLICY_HPP
#define MLPACK_CORE_DATA_FILE_READER_POLICY_HPP

#include <mlpack/core/data/file_reader/reader_exceptions.hpp>

namespace mlpack{

namespace io{

#include <limits>

template<char ... TrimCharList>
struct TrimChars
{
 public:
  static void Trim(char*&strBegin, char*&strEnd)
  {
    while(strBegin != strEnd && IsTrimChar(*strBegin, TrimCharList...))
    {
      ++strBegin;
    }
    while(strBegin != strEnd && IsTrimChar(*(strEnd-1), TrimCharList...))
    {
      --strEnd;
    }
    *strEnd = '\0';
  }

 private:
  constexpr static bool IsTrimChar(char)
  {
    return false;
  }

  template<class ...OtherTrimChars>
  constexpr static bool IsTrimChar(char c, char trimChar, OtherTrimChars...otherTrimChars)
  {
    return c == trimChar || IsTrimChar(c, otherTrimChars...);
  }
};


struct NoComment
{
  static bool IsComment(const char*)
  {
    return false;
  }
};

template<char ... CommentStartCharList>
struct SingleLineComment
{ 
 public:
  static bool IsComment(const char* line)
  {
    return IsCommentStartChar(*line, CommentStartCharList...);
  }

 private:
  constexpr static bool IsCommentStartChar(char)
  {
    return false;
  }

  template<class ...OtherCommentStartChars>
  constexpr static bool IsCommentStartChar(char c, char commentstartchar,
                                           OtherCommentStartChars...othercommentstartchars)
  {
    return c == commentstartchar || IsCommentStartChar(c, othercommentstartchars...);
  }
};

struct EmptyLineComment
{
  static bool IsComment(const char* line)
  {
    if(*line == '\0')
    {
      return true;
    }

    while(*line == ' ' || *line == '\t')
    {
      ++line;
      if(*line == 0){
        return true;
      }
    }
    return false;
  }
};

template<char ... CommentStartCharList>
struct SingleAndEmptyLineComment
{
  static bool IsComment(const char *line)
  {
    return SingleLineComment<CommentStartCharList...>::IsComment(line) ||
        EmptyLineComment::IsComment(line);
  }
};

template<char sep>
struct NoQuoteEscape
{
  static const char* FindNextColumnEnd(const char*col_begin)
  {
    while(*col_begin != sep && *col_begin != '\0')
    {
      ++col_begin;
    }
    return col_begin;
  }

  static void unescape(char*&, char*&)
  {

  }
};

template<char sep, char quote>
struct DoubleQuoteEscape
{
  static const char* FindNextColumnEnd(const char*col_begin)
  {
    while(*col_begin != sep && *col_begin != '\0')
      if(*col_begin != quote)
      {
        ++col_begin;
      }else{
        do{
          ++col_begin;
          while(*col_begin != quote){
            if(*col_begin == '\0')
              throw error::EscapedStringNotClosed();
            ++col_begin;
          }
          ++col_begin;
        }while(*col_begin == quote);
      }
    return col_begin;
  }

  static void unescape(char *&col_begin, char *&col_end)
  {
    if(col_end - col_begin >= 2){
      if(*col_begin == quote && *(col_end-1) == quote){
        ++col_begin;
        --col_end;
        char *out = col_begin;
        for(char*in = col_begin; in!=col_end; ++in){
          if(*in == quote && *(in+1) == quote){
            ++in;
          }
          *out = *in;
          ++out;
        }
        col_end = out;
        *col_end = '\0';
      }
    }
  }
};

struct ThrowOnOverflow
{
  template<class T>
  static void OnOverflow(T&)
  {
    throw error::IntegerOverflow();
  }

  template<class T>
  static void OnUnderflow(T&)
  {
    throw error::IntegerUnderflow();
  }
};

struct IgnoreOverflow
{
  template<class T>
  static void OnOverFlow(T&){}

  template<class T>
  static void OnUnderFlow(T&){}
};

struct SetToMaxOnOverflow
{
  template<class T>
  static void OnOverFlow(T&x)
  {
    x = std::numeric_limits<T>::max();
  }

  template<class T>
  static void OnUnderFlow(T&x)
  {
    x = std::numeric_limits<T>::min();
  }
};

} //namespace io

} //namespace mlpack

#endif
