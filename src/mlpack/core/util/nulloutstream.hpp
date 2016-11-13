/**
 * @file nulloutstream.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Definition of the NullOutStream class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_NULLOUTSTREAM_HPP
#define MLPACK_CORE_UTIL_NULLOUTSTREAM_HPP

#include <iostream>
#include <streambuf>
#include <string>

namespace mlpack {
namespace util {

/**
 * Used for Log::Debug when not compiled with debugging symbols.  This class
 * does nothing and should be optimized out entirely by the compiler.
 */
class NullOutStream
{
 public:
  /**
   * Does nothing.
   */
  NullOutStream() { }

  /**
   * Does nothing.
   */
  NullOutStream(const NullOutStream& /* other */) { }

  //! Does nothing.
  NullOutStream& operator<<(bool) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(short) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(unsigned short) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(int) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(unsigned int) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(long) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(unsigned long) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(float) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(double) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(long double) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(void*) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(const char*) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(std::string&) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(std::streambuf*) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(std::ostream& (*) (std::ostream&)) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(std::ios& (*) (std::ios&)) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(std::ios_base& (*) (std::ios_base&))
  { return *this; }

  //! Does nothing.
  template<typename T>
  NullOutStream& operator<<(const T&) { return *this; }
};

} // namespace util
} // namespace mlpack

#endif
