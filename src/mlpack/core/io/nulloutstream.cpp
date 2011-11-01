/**
 * @file nulloutstream.cpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Implementation of NullOutStream functions.
 */
#include "nulloutstream.hpp"

using namespace mlpack::io;

NullOutStream::NullOutStream()
{ /* Nothing to do */ }

NullOutStream::NullOutStream(const NullOutStream& other)
{ /* Nothing to do */ }

NullOutStream& NullOutStream::operator<< (bool val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (short val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (unsigned short val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (int val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (unsigned int val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (long val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (unsigned long val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (float val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (double val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (long double val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (void* val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (std::string& str)
{ return *this; }

NullOutStream& NullOutStream::operator<< (const char* str)
{ return *this; }

NullOutStream& NullOutStream::operator<< (std::streambuf* val)
{ return *this; }

NullOutStream& NullOutStream::operator<< (std::ostream& (*pf) (std::ostream&))
{
  return *this;
}

NullOutStream& NullOutStream::operator<< (std::ios& (*pf) (std::ios&))
{
  return *this;
}

NullOutStream& NullOutStream::operator<< (std::ios_base& (*pf) (std::ios_base&))
{
  return *this;
}


