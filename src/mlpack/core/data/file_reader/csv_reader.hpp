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

#ifndef MLPACK_CORE_DATA_FILE_READER_CSV_READER_HPP
#define MLPACK_CORE_DATA_FILE_READER_CSV_READER_HPP

#include "line_reader.hpp"
#include "parser.hpp"

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

#include <numeric>

namespace mlpack{

namespace io{

template<class TrimPolicy = TrimChars<' ', '\t'>,
         class QuotePolicy = NoQuoteEscape<','>,
         class OverflowPolicy = ThrowOnOverflow,
         class CommentPolicy = NoComment
         >
class CSVReader{
 public:
  CSVReader() = delete;
  CSVReader(const CSVReader&) = delete;
  CSVReader&operator=(const CSVReader&);

  template<class ...Args>
  explicit CSVReader(size_t column_count, Args&&...args) :
    in(std::forward<Args>(args)...),
    columncount(column_count),
    columnNames(column_count),
    colOrder(column_count),
    row(column_count, nullptr)
  {
    std::iota(std::begin(colOrder), std::end(colOrder), 0);
    for(size_t i=1; i<=column_count; ++i){
      columnNames[i-1] = "col" + std::to_string(i);
    }
  }

  static void FileDimension(std::string const &fileName,
                            std::string const &separators,
                            size_t &rows, size_t &cols)
  {
    LineReader reader(fileName);
    rows = 0; cols = 0;
    char *line = reader.NextLine();
    if(line){
      using Tokenizer = boost::tokenizer<boost::escaped_list_separator<char>>;
      boost::escaped_list_separator<char> sep("\\", separators, "\"");
      std::string buffer(line, std::strlen(line));
      Tokenizer tok(buffer, sep);{
      for(Tokenizer::iterator i = tok.begin(); i != tok.end(); ++i)
        ++cols;
      }

      ++rows;
      while(reader.NextLine()){
        ++rows;
      }
    }
  }

  template<typename OutIter>
  bool ReadRow(OutIter begin, OutIter end)
  {
    try{
      try{
        char *line = PruneComment();
        if(line){
          ParseLine<TrimPolicy, QuotePolicy>(line, &row[0], colOrder);
        }else{
          return false;
        }

        ParseHelper(begin, end);
      }catch(error::WithFileName &err){
        err.FileName(in.TruncatedFileName());
        throw;
      }
    }catch(error::WithFileLine&err){
      err.FileLine(in.FileLine());
      throw;
    }

    return true;
  }

  template<typename Container>
  bool ReadRow(Container &colVals)
  {
    return ReadRow(std::begin(colVals), std::end(colVals));
  }

  template<class ...ColType>
  bool ReadRow(ColType& ...cols)
  {
    try{
      try{
        char *line = PruneComment();
        if(line){
          ParseLine<TrimPolicy, QuotePolicy>
              (line, &row[0], colOrder);
          ParseHelper(0, cols...);
        }else{
          return false;
        }
      }catch(error::WithFileName &err){
        err.FileName(in.TruncatedFileName());
        throw;
      }
    }catch(error::WithFileLine&err){
      err.FileLine(in.FileLine());
      throw;
    }

    return true;
  }

  char* NextLine(){
    return in.NextLine();
  }

  void FileName(const std::string&file_name)
  {
    in.FileName(file_name);
  }

  void FileName(const char*file_name)
  {
    in.FileName(file_name);
  }

  const char* TruncatedFileName()const
  {
    return in.TruncatedFileName();
  }

  void FileLine(unsigned file_line)
  {
    in.FileLine(file_line);
  }

  size_t FileLine()const
  {
    return in.FileLine();
  }

 private:
  void ParseHelper(std::size_t){}

  template<class T, class ...ColType>
  void ParseHelper(std::size_t r, T&t, ColType&...cols)
  {
    if(row[r]){
      try{
        try{
          Parse<OverflowPolicy>(row[r], t);
        }catch(error::WithColumnContent&err){
          err.ColumnContent(row[r]);
          throw;
        }
      }catch(error::WithColumnName&err){
        err.ColumnName(columnNames[r].c_str());
        throw;
      }
    }
    ParseHelper(r+1, cols...);
  }

  template<typename OutIter>
  void ParseHelper(OutIter begin, OutIter end)
  {
    std::size_t r = 0;
    try{
      try{
        while(begin != end){
          if(row[r]){
            Parse<OverflowPolicy>(row[r++], *begin);
          }
          ++begin;
        }
      }catch(error::WithColumnContent&err){
        err.ColumnContent(row[r]);
        throw;
      }
    }catch(error::WithColumnName&err){
      err.ColumnName(columnNames[r].c_str());
      throw;
    }
  }

  char* PruneComment()
  {
    char *line;
    do{
      line = in.NextLine();
      if(!line){
        return nullptr;
      }
    }while(CommentPolicy::IsComment(line));

    return line;
  }

  LineReader in;

  size_t columncount;
  std::vector<std::string> columnNames;
  std::vector<int> colOrder;
  std::vector<char*> row;
};

}//namespace io

}//namespace mlpack

#endif

