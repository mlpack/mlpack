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

#ifndef READER_EXCEPTIONS_HPP
#define READER_EXCEPTIONS_HPP

#include <boost/format.hpp>

#include <exception>
#include <cassert>
#include <cerrno>
#include <istream>
#include <string>

namespace mlpack{

namespace io{

namespace error{

struct Base : std::exception
{
  virtual void FormatErrorMessage()const = 0;

  const char* what()const throw()
  {
    FormatErrorMessage();
    return errorMessageBuffer.c_str();
  }

  mutable std::string errorMessageBuffer;
};

struct WithFileName
{
  WithFileName(){
  }

  void FileName(const char*file_name)
  {
    fileName = file_name;
  }

  std::string fileName;
};

struct WithFileLine
{
  WithFileLine() :
  fileLine(0)
  {
  }

  void FileLine(size_t fileLine)
  {
    this->fileLine = fileLine;
  }

  size_t fileLine;
};

struct CanNotOpenFile :
    Base,
    WithFileName
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer = "Can not open file \"" + fileName + "\"";
  }
};

struct LineLengthLimitExceeded :
    Base,
    WithFileName,
    WithFileLine
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer = "Line length " +std::to_string(fileLine) +
        " in file \"" + fileName + "\" exceeds the maximum length of 2^24-1.";
  }
};

struct WithColumnName
{
  void ColumnName(const char* columnName)
  {
    this->columnName = columnName;
  }

  std::string columnName;
};

struct WithColumnContent
{
  void ColumnContent(const char *columnContent){
    this->columnContent = columnContent;
  }

  std::string columnContent;
};


struct ExtraColumnInHeader :
    Base,
    WithFileName,
    WithColumnName
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer = std::string("Extra column \"") + columnName +
        "\" in header of file \"" + fileName + "\"";
  }
};

struct MissingColumnInHeader :
    Base,
    WithFileName,
    WithColumnName{
  void FormatErrorMessage()const
  {
    errorMessageBuffer = (boost::format("Missing column \"%1%\" in header of file \"%2%\".")
        % columnName % fileName).str();
  }
};

struct DuplicatedColumnInHeader :
    Base,
    WithFileName,
    WithColumnName
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer= (boost::format("Duplicated column \"%1%\" in header of file \"%2%\".")
                         % columnName % fileName).str();
  }
};

struct HeaderMissing :
    Base,
    WithFileName
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer= (boost::format("Header missing in file \"%1%\".")
                         % fileName).str();
  }
};

struct TooFewColumns :
    Base,
    WithFileName,
    WithFileLine
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer= (boost::format("Too few columns in line %1% in file \"%2%\".")
                         % fileLine % fileName).str();
  }
};

struct TooManyColumns :
    Base,
    WithFileName,
    WithFileLine
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer= (boost::format("Too many columns in line %1% in file \"%2%\".")
                         % fileLine % fileName).str();
  }
};

struct EscapedStringNotClosed :
    Base,
    WithFileName,
    WithFileLine
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer= (boost::format("Escaped string was not closed in line %1% in file \"%2%\".")
                         % fileLine % fileName).str();
  }
};

struct IntegerMustBePositive :
    Base,
    WithFileName,
    WithFileLine,
    WithColumnName,
    WithColumnContent
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer= (boost::format("The integer \"%1%\" must be positive or 0 in "
                                       "column \"%2%\" in file \"%3%\" in line \"%4%\".")
                         % columnContent % columnName %
                         fileName % fileLine).str();
  }
};

struct NoDigit :
    Base,
    WithFileName,
    WithFileLine,
    WithColumnName,
    WithColumnContent
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer= (boost::format("The integer \"%1%\" contains an invalid digit in column "
                                       "\"%2%\" in file \"%3%\" in line \"%4%\".")
                         % columnContent % columnName %
                         fileName % fileLine).str();
  }
};

struct IntegerOverflow :
    Base,
    WithFileName,
    WithFileLine,
    WithColumnName,
    WithColumnContent
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer= (boost::format("The integer \"%1%\" overflows in column \"%2%\" "
                                       "in file \"%3%\" in line \"%4%\".")
                         % columnContent % columnName %
                         fileName % fileLine).str();
  }
};

struct IntegerUnderflow :
    Base,
    WithFileName,
    WithFileLine,
    WithColumnName,
    WithColumnContent
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer= (boost::format("The integer \"%1%\" underflows in column \"%2%\" "
                                       "in file \"%3%\" in line \"%4%\".")
                         % columnContent % columnName %
                         fileName % fileLine).str();
  }
};

struct InvalidSingleCharacter :
    Base,
    WithFileName,
    WithFileLine,
    WithColumnName,
    WithColumnContent
{
  void FormatErrorMessage()const
  {
    errorMessageBuffer= (boost::format("The content \"%1%\" of column \"%2%\" "
                                       "in file \"%3%\", in line \"%4%\" is not a "
                                       "single character.")
                         % columnContent % columnName %
                         fileName % fileLine).str();
  }
};

}//namespace error

} //namespace io

} //namespace mlpack

#endif
