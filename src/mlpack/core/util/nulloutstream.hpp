/**
 * @file nulloutstream.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Definition of the NullOutStream class.
 */
#ifndef __MLPACK_CORE_IO_NULLOUTSTREAM_HPP
#define __MLPACK_CORE_IO_NULLOUTSTREAM_HPP

#include <iostream>
#include <streambuf>
#include <string>

namespace mlpack {
namespace io {

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
  NullOutStream(const NullOutStream& other) { }

  //! Does nothing.
  NullOutStream& operator<<(bool val) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(short val) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(unsigned short val) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(int val) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(unsigned int val) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(long val) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(unsigned long val) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(float val) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(double val) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(long double val) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(void* val) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(const char* str) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(std::string& str) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(std::streambuf* sb) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(std::ostream& (*pf) (std::ostream&))
  { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(std::ios& (*pf) (std::ios&)) { return *this; }
  //! Does nothing.
  NullOutStream& operator<<(std::ios_base& (*pf) (std::ios_base&))
  { return *this; }

  //! Does nothing.
  template<typename T>
  NullOutStream& operator<<(T s)
  { return *this; }
};

} // namespace io
} // namespace mlpack

#endif
