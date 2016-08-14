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

#ifndef MLPACK_CORE_DATA_FILE_READER_PARSER_HPP
#define MLPACK_CORE_DATA_FILE_READER_PARSER_HPP

#include <mlpack/core/data/file_reader/policy.hpp>

#include <limits>
#include <vector>

namespace mlpack{

namespace io{

template<class QuotePolicy>
void ChopNextColumn(char*& line, char*& colBegin, char*& colEnd)
{
  assert(line != nullptr);

  colBegin = line;
  // the col_begin + (... - col_begin) removes the constness
  colEnd = colBegin + (QuotePolicy::FindNextColumnEnd(colBegin) - colBegin);

  if(*colEnd == '\0'){
    line = nullptr;
  }else{
    *colEnd = '\0';
    line = colEnd + 1;
  }
}

template<class TrimPolicy, class QuotePolicy>
void ParseLine(
    char *line,
    char **sortedCol,
    const std::vector<int> &colOrder
    )
{
  for(std::size_t i=0; i<colOrder.size(); ++i){
    if(line == nullptr){
      throw error::TooFewColumns();
    }
    char*col_begin, *col_end;
    ChopNextColumn<QuotePolicy>(line, col_begin, col_end);

    if(colOrder[i] != -1){
      TrimPolicy::Trim(col_begin, col_end);
      QuotePolicy::unescape(col_begin, col_end);

      sortedCol[colOrder[i]] = col_begin;
    }
  }
  if(line != nullptr)
    throw error::TooManyColumns();
}

template<class OverFlowPolicy>
void Parse(char *col, char &x){
  if(!*col)
    throw error::InvalidSingleCharacter();
  x = *col;
  ++col;
  if(*col)
    throw error::InvalidSingleCharacter();
}

template<class OverFlowPolicy>
void Parse(char *col, std::string &x){
  x = col;
}

template<class OverFlowPolicy>
void Parse(char* col, const char*& x){
  x = col;
}

template<class OverFlowPolicy>
void Parse(char*col, char*& x){
  x = col;
}

template<class OverFlowPolicy, class T>
void ParseUnsignedInteger(const char *col, T &x){
  x = 0;
  while(*col != '\0'){
    if('0' <= *col && *col <= '9'){
      T y = *col - '0';
      if(x > (std::numeric_limits<T>::max()-y)/10){        
        OverFlowPolicy::OnOverFlow(x);
        return;
      }
      x = 10*x+y;
    }else{
      throw error::NoDigit();
    }
    ++col;
  }
}

template<class OverFlowPolicy>void Parse(char *col, unsigned char &x)
{ParseUnsignedInteger<OverFlowPolicy>(col, x);}
template<class OverFlowPolicy>void Parse(char *col, unsigned short &x)
{ParseUnsignedInteger<OverFlowPolicy>(col, x);}
template<class OverFlowPolicy>void Parse(char *col, unsigned int &x)
{ParseUnsignedInteger<OverFlowPolicy>(col, x);}
template<class OverFlowPolicy>void Parse(char *col, unsigned long &x)
{ParseUnsignedInteger<OverFlowPolicy>(col, x);}
template<class OverFlowPolicy>void Parse(char *col, unsigned long long &x)
{ParseUnsignedInteger<OverFlowPolicy>(col, x);}

template<class OverFlowPolicy, class T>
void ParseSignedInteger(const char *col, T &x){
  if(*col == '-'){
    ++col;

    x = 0;
    while(*col != '\0'){
      if('0' <= *col && *col <= '9'){
        T y = *col - '0';
        if(x < (std::numeric_limits<T>::min()+y)/10){
          OverFlowPolicy::OnOverFlow(x);
          return;
        }
        x = 10*x-y;
      }else
        throw error::NoDigit();
      ++col;
    }
    return;
  }else if(*col == '+'){
    ++col;
  }
  ParseUnsignedInteger<OverFlowPolicy>(col, x);
}

template<class OverFlowPolicy>void Parse(char *col, signed char &x)
{ParseSignedInteger<OverFlowPolicy>(col, x);}
template<class OverFlowPolicy>void Parse(char *col, signed short &x)
{ParseSignedInteger<OverFlowPolicy>(col, x);}
template<class OverFlowPolicy>void Parse(char *col, signed int &x)
{ParseSignedInteger<OverFlowPolicy>(col, x);}
template<class OverFlowPolicy>void Parse(char *col, signed long &x)
{ParseSignedInteger<OverFlowPolicy>(col, x);}
template<class OverFlowPolicy>void Parse(char *col, signed long long &x)
{ParseSignedInteger<OverFlowPolicy>(col, x);}

template<class T>
void ParseFloat(const char *col, T &x){
  bool is_neg = false;
  if(*col == '-'){
    is_neg = true;
    ++col;
  }else if(*col == '+'){
    ++col;
  }

  x = 0;
  while('0' <= *col && *col <= '9'){
    int const y = *col - '0';
    x *= 10;
    x += static_cast<T>(y);
    ++col;
  }

  if(*col == '.'|| *col == ','){
    ++col;
    T pos = 1;
    while('0' <= *col && *col <= '9'){
      pos /= 10;
      int const y = *col - '0';
      ++col;
      x += y*pos;
    }
  }

  if(*col == 'e' || *col == 'E'){
    ++col;
    int e;

    ParseSignedInteger<SetToMaxOnOverflow>(col, e);

    if(e != 0){
      T base;
      if(e < 0){
        base = static_cast<T>(0.1);
        e = -e;
      }else{
        base = 10;
      }

      while(e != 1){
        if((e & 1) == 0){
          base = base*base;
          e >>= 1;
        }else{
          x *= base;
          --e;
        }
      }
      x *= base;
    }
  }else{
    if(*col != '\0'){
      throw error::NoDigit();
    }
  }

  if(is_neg)
    x = -x;
}

template<class OverFlowPolicy> void Parse(char *col, float &x) { ParseFloat(col, x); }
template<class OverFlowPolicy> void Parse(char *col, double &x) { ParseFloat(col, x); }
template<class OverFlowPolicy> void Parse(char *col, long double &x) { ParseFloat(col, x); }

template<class OverFlowPolicy, class T>
void Parse(char*, T&){
  // GCC evalutes "false" when reading the template and
  // "sizeof(T)!=sizeof(T)" only when instantiating it. This is why
  // this strange construct is used.
  static_assert(sizeof(T)!=sizeof(T),
                "Can not parse this type. Only buildin integrals, floats, char, "
                "char*, const char* and std::string are supported");
}

} //namespace io

} //namespace mlpack

#endif
