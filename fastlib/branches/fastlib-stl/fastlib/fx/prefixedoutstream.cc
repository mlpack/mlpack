#include "prefixedoutstream.h"

#include <string>
#include <iostream>
#include <streambuf>
#include <string.h>
#include <stdlib.h>

using namespace mlpack::io;

PrefixedOutStream& PrefixedOutStream::operator<< (bool val) {
  BaseLogic<bool>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (short val) {
  BaseLogic<short>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (unsigned short val) {
 BaseLogic<unsigned short>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (int val) {
  BaseLogic<int>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (unsigned int val) {
  BaseLogic<unsigned int>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (long val) {
  BaseLogic<long>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (unsigned long val) {
  BaseLogic<unsigned long>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (float val) {
  BaseLogic<float>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (double val) {
  BaseLogic<double>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (long double val) {
  BaseLogic<long double>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (void* val) {
  BaseLogic<void*>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (const char* str) {
  BaseLogic<const char*>(str);
  
  // No need to worry about platform-independent newline characters; because
  // stdout and stderr are text-mode streams, they do the relevant conversions
  // automatically.
  if (strstr(str, "\n") != NULL) {
    carriageReturned = true;
    if (fatal) // Terminate application.
      exit(1);
  }

  return *this;
}


PrefixedOutStream& PrefixedOutStream::operator<< (std::string& str) {
  
  BaseLogic<std::string&>(str);
  
  // No need to worry about platform-independent newline characters; because
  // stdout and stderr are text-mode streams, they do the relevant conversions
  // automatically.
  if (str.find("\n") != std::string::npos) {
    carriageReturned = true;
    if (fatal) // Terminate application.
      exit(1);
  }

  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (std::streambuf* sb) {
  destination << sb;
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< 
    (std::ostream& (*pf) (std::ostream&)) {
  // We don't want to prefix on what will show up as empty lines. 
  destination << pf;
  carriageReturned = true;
  if (fatal) // Terminate application.
    exit(1);

  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< (std::ios& (*pf) (std::ios&)) {
  
  BaseLogic<std::ios& (*) (std::ios&)>(pf);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<< 
    (std::ios_base& (*pf) (std::ios_base&)) {
      
  BaseLogic<std::ios_base& (*) (std::ios_base&)>(pf);
  return *this;
}
 
