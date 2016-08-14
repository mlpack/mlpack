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

#ifndef MLPACK_CORE_DATA_FILE_READER_LINE_READER_HPP
#define MLPACK_CORE_DATA_FILE_READER_LINE_READER_HPP

#include <mlpack/core/data/file_reader/reader_exceptions.hpp>

#include <cstring>
#include <algorithm>
#include <fstream>
#include <memory>
#include <utility>
#include <cstdio>
#include <vector>

namespace mlpack{

namespace io{

namespace detail{

class OwningStdIOByteSourceBase
{
 public:
  explicit OwningStdIOByteSourceBase(std::unique_ptr<std::ifstream> file):
    file(std::move(file)){
    std::ios_base::sync_with_stdio(false);
  }

  int Read(char *buffer, const size_t size){
    file->read(buffer, size);
    return static_cast<int>(file->gcount());
  }

  ~OwningStdIOByteSourceBase()
  {
    std::ios_base::sync_with_stdio(true);
  }

 private:
  std::unique_ptr<std::ifstream> file;
};

class SynchronousReader{
 public:
  void Init(std::unique_ptr<OwningStdIOByteSourceBase> arg_byte_source){
    byteSource = std::move(arg_byte_source);
  }

  bool IsValid()const{
    return byteSource != nullptr;
  }

  void PrepareRead(char*arg_buffer, int arg_desired_byte_count){
    buffer = arg_buffer;
    desiredByteCount = arg_desired_byte_count;
  }

  int FinishRead(){
    return byteSource->Read(buffer, desiredByteCount);
  }

 private:
  std::unique_ptr<OwningStdIOByteSourceBase> byteSource;
  char *buffer;
  int desiredByteCount;
};

} //namespace details

class LineReader{
 public:
  LineReader() = delete;
  LineReader(const LineReader&) = delete;
  LineReader&operator=(const LineReader&) = delete;

  explicit LineReader(const char *fileName){
    FileName(fileName);
    Init(OpenFile(fileName));
  }

  explicit LineReader(const std::string &fileName){
    FileName(fileName.c_str());
    Init(OpenFile(fileName.c_str()));
  }

  void FileName(const std::string &fileName){
    FileName(fileName.c_str());
  }

  void FileName(const char* fileName)
  {
    this->fileName = fileName;
  }

  const char* TruncatedFileName()const
  {
    return fileName.c_str();
  }

  void FileLine(size_t fileLine)
  {
    this->fileLine = fileLine;
  }

  size_t FileLine()const
  {
    return fileLine;
  }

  int LineLength() const
  {
    return lineLength;
  }

  char* NextLine()
  {
    if(dataBegin == dataEnd){
      return nullptr;
    }

    ++fileLine;

    assert(dataBegin < dataEnd);
    assert(dataEnd <= blockLen*2);

    if(dataBegin >= blockLen){
      //first block has been processed, copy second block to first block
      std::memcpy(&buffer[0], &buffer[0]+blockLen, blockLen);
      dataBegin -= blockLen;
      dataEnd -= blockLen;
      //if the file >= 2 blockLen, that means we need to read more data
      if(reader.IsValid())
      {
        dataEnd += reader.FinishRead();
        std::memcpy(&buffer[0]+blockLen, &buffer[0]+2*blockLen, blockLen);
        reader.PrepareRead(&buffer[0] + 2*blockLen, blockLen);
      }
    }

    int lineEnd = dataBegin;
    while(buffer[lineEnd] != '\n' && lineEnd != dataEnd){
      ++lineEnd;
    }

    if(lineEnd - dataBegin + 1 > blockLen){
      error::LineLengthLimitExceeded err;
      err.FileName(fileName.c_str());
      err.FileLine(fileLine);
      throw err;
    }

    if(buffer[lineEnd] == '\n'){
      buffer[lineEnd] = '\0';
    }else{
      // some files are missing the newline at the end of the
      // last line
      ++dataEnd;
      buffer[lineEnd] = '\0';
    }

    // handle windows \r\n-line breaks
    if(lineEnd != dataBegin && buffer[lineEnd-1] == '\r'){
      buffer[lineEnd-1] = '\0';
    }

    char *ret = &buffer[0] + dataBegin;
    lineLength = lineEnd - dataBegin - 1;
    dataBegin = lineEnd + 1;
    return ret;
  }

 private:
  //blockLen equal to the limit of one line
  static constexpr int blockLen = 1<<24;

  detail::SynchronousReader reader;
  std::vector<char> buffer;
  int dataBegin;
  int dataEnd;
  int lineLength;

  std::string fileName;
  size_t fileLine;

  static std::unique_ptr<detail::OwningStdIOByteSourceBase> OpenFile(const char *file_name)
  {
    std::unique_ptr<std::ifstream> file(new std::ifstream(file_name, std::ios::binary));
    if(!file->is_open()){
      error::CanNotOpenFile err;
      err.FileName(file_name);
      throw err;
    }

    return std::unique_ptr<detail::OwningStdIOByteSourceBase>
        (new detail::OwningStdIOByteSourceBase(std::move(file)));
  }

  void Init(std::unique_ptr<detail::OwningStdIOByteSourceBase> byteSource)
  {
    fileLine = 0;
    lineLength = 0;

    //Allocate 48MBytes to store char of files
    //First block store the string we want to handle
    //Second block store the extra string to handle
    //after the First block is consumed
    //Third block is the "prepare block", use to read more
    //data from the file
    buffer.resize(3*blockLen);
    dataBegin = 0;
    dataEnd = byteSource->Read(&buffer[0], 2*blockLen);

    //Ignore UTF-8 BOM
    if(dataEnd >= 3 && buffer[0] == '\xEF' && buffer[1] == '\xBB' && buffer[2] == '\xBF'){
      dataBegin = 3;
    }

    //If the data of file is >= 2*blockLen, we need to do
    //the prepare of reading more data
    if(dataEnd == 2*blockLen){
      reader.Init(std::move(byteSource));
      reader.PrepareRead(&buffer[0] + 2*blockLen, blockLen);
    }
  }
};

} //namespace io

} //namespace mlpack

#endif
