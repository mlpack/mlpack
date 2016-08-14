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

#ifndef MLPACK_CORE_DATA_FILE_READER_READER_EXCEPTIONS_HPP
#define MLPACK_CORE_DATA_FILE_READER_READER_EXCEPTIONS_HPP

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
    return errorMessageBuffer;
  }

  mutable char errorMessageBuffer[256];
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

struct WithErrno
{
  WithErrno(){
    errnoValue = 0;
  }

  void Errno(int errno_value)
  {
    this->errnoValue = errnoValue;
  }

  int errnoValue;
};

struct CanNotOpenFile :
    Base,
    WithFileName,
    WithErrno
{
  void FormatErrorMessage()const
  {
    if(errnoValue != 0){
      std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                    "Can not open file \"%s\" because \"%s\"."
                    ,&fileName[0], std::strerror(errnoValue));
    }else{
      std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                    "Can not open file \"%s\"."
                    ,&fileName[0]);
    }
  }
};

struct LineLengthLimitExceeded :
    Base,
    WithFileName,
    WithFileLine
{
  void FormatErrorMessage()const
  {
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "Line length %d in file \"%s\" exceeds the maximum length of 2^24-1."
                  , fileLine, &fileName[0]);
  }
};

struct WithColumnName
{
  WithColumnName()
  {
    std::fill(std::begin(columnName), std::end(columnName), 0);
  }

  void ColumnName(const char* columnName)
  {
    std::strncpy(this->columnName, columnName, maxColumnNameLength);
    this->columnName[maxColumnNameLength] = '\0';
  }

  static constexpr int maxColumnNameLength = 63;
  char columnName[maxColumnNameLength+1];
};

struct WithColumnContent
{
  WithColumnContent(){
    std::memset(columnContent, 0, maxColumnContentLength+1);
  }

  void ColumnContent(const char *columnContent){
    std::strncpy(this->columnContent, columnContent, maxColumnContentLength);
    this->columnContent[maxColumnContentLength] = '\0';
  }

  static constexpr int maxColumnContentLength = 63;
  char columnContent[maxColumnContentLength+1];
};


struct ExtraColumnInHeader :
    Base,
    WithFileName,
    WithColumnName
{
  void FormatErrorMessage()const
  {
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "Extra column \"%s\" in header of file \"%s\"."
                  , columnName, &fileName[0]);
  }
};

struct MissingColumnInHeader :
    Base,
    WithFileName,
    WithColumnName{
  void FormatErrorMessage()const
  {
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "Missing column \"%s\" in header of file \"%s\"."
                  , columnName, &fileName[0]);
  }
};

struct DuplicatedColumnInHeader :
    Base,
    WithFileName,
    WithColumnName
{
  void FormatErrorMessage()const
  {
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "Duplicated column \"%s\" in header of file \"%s\"."
                  , columnName, &fileName[0]);
  }
};

struct HeaderMissing :
    Base,
    WithFileName
{
  void FormatErrorMessage()const
  {
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "Header missing in file \"%s\"."
                  , &fileName[0]);
  }
};

struct TooFewColumns :
    Base,
    WithFileName,
    WithFileLine
{
  void FormatErrorMessage()const
  {
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "Too few columns in line %d in file \"%s\"."
                  , fileLine, &fileName[0]);
  }
};

struct TooManyColumns :
    Base,
    WithFileName,
    WithFileLine
{
  void FormatErrorMessage()const
  {
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "Too many columns in line %d in file \"%s\"."
                  , fileLine, &fileName[0]);
  }
};

struct EscapedStringNotClosed :
    Base,
    WithFileName,
    WithFileLine
{
  void FormatErrorMessage()const
  {
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "Escaped string was not closed in line %d in file \"%s\"."
                  , fileLine, &fileName[0]);
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
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "The integer \"%s\" must be positive or 0 in column \"%s\" in file \"%s\" in line \"%d\"."
                  , columnContent, columnName, &fileName[0], fileLine);
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
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "The integer \"%s\" contains an invalid digit in column \"%s\" in file \"%s\" in line \"%d\"."
                  , columnContent, columnName, &fileName[0], fileLine);
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
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "The integer \"%s\" overflows in column \"%s\" in file \"%s\" in line \"%d\"."
                  , columnContent, columnName, &fileName[0], fileLine);
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
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "The integer \"%s\" underflows in column \"%s\" in file \"%s\" in line \"%d\"."
                  , columnContent, columnName, &fileName[0], fileLine);
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
    std::snprintf(errorMessageBuffer, sizeof(errorMessageBuffer),
                  "The content \"%s\" of column \"%s\" in file \"%s\" in line \"%d\" is not a single character."
                  , columnContent, columnName, &fileName[0], fileLine);
  }
};

}//namespace error

} //namespace io

} //namespace mlpack

#endif
