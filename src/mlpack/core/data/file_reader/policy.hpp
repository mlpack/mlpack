#ifndef POLICY_HPP
#define POLICY_HPP

namespace mlpack{

namespace io{

//typedef unsigned ignore_column;
//static const ignore_column ignore_no_column = 0;
//static const ignore_column ignore_extra_column = 1;
//static const ignore_column ignore_missing_column = 2;

#include <limits>

template<char ... TrimCharList>
struct TrimChars
{
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

public:
  static bool IsComment(const char* line)
  {
    return IsCommentStartChar(*line, CommentStartCharList...);
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
