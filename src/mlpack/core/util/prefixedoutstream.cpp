/**
 * @file prefixedoutstream.cpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Implementation of PrefixedOutStream methods.
 */
#include <string>
#include <iostream>
#include <streambuf>
#include <string.h>
#include <stdlib.h>

#include "prefixedoutstream.hpp"

using namespace mlpack::util;

/**
 * These are all necessary because gcc's template mechanism does not seem smart
 * enough to figure out what I want to pass into operator<< without these.  That
 * may not be the actual case, but it works when these is here.
 */

PrefixedOutStream& PrefixedOutStream::operator<<(bool val)
{
  BaseLogic<bool>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(short val)
{
  BaseLogic<short>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(unsigned short val)
{
 BaseLogic<unsigned short>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(int val)
{
  BaseLogic<int>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(unsigned int val)
{
  BaseLogic<unsigned int>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(long val)
{
  BaseLogic<long>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(unsigned long val)
{
  BaseLogic<unsigned long>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(float val)
{
  BaseLogic<float>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(double val)
{
  BaseLogic<double>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(long double val)
{
  BaseLogic<long double>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(void* val)
{
  BaseLogic<void*>(val);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(const char* str)
{
  BaseLogic<const char*>(str);
  return *this;
}


PrefixedOutStream& PrefixedOutStream::operator<<(std::string& str)
{
  BaseLogic<std::string>(str);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(std::streambuf* sb)
{
  BaseLogic<std::streambuf*>(sb);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(
    std::ostream& (*pf)(std::ostream&))
{
  BaseLogic<std::ostream& (*)(std::ostream&)>(pf);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(std::ios& (*pf)(std::ios&))
{
  BaseLogic<std::ios& (*)(std::ios&)>(pf);
  return *this;
}

PrefixedOutStream& PrefixedOutStream::operator<<(
    std::ios_base& (*pf) (std::ios_base&))
{
  BaseLogic<std::ios_base& (*)(std::ios_base&)>(pf);
  return *this;
}
