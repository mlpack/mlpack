/**
 * @file nulloutstream.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Definition of the NullOutStream class.
 */
#ifndef __MLPACK_CORE_IO_NULL_OUT_STREAM_HPP
#define __MLPACK_CORE_IO_NULL_OUT_STREAM_HPP

#include <iostream>
#include <streambuf>
#include <string>

namespace mlpack {
namespace io {

/**
 * Used for Log::Debug when not compiled with debugging symbols.  This class
 * does nothing and should be optimized out entirely by the compiler.
 */
class NullOutStream {
 public:
  /**
   * Does nothing.
   */
  NullOutStream();

  /**
   * Does nothing.
   */
  NullOutStream(const NullOutStream& other);

  //! Does nothing.
  NullOutStream& operator<<(bool val);
  //! Does nothing.
  NullOutStream& operator<<(short val);
  //! Does nothing.
  NullOutStream& operator<<(unsigned short val);
  //! Does nothing.
  NullOutStream& operator<<(int val);
  //! Does nothing.
  NullOutStream& operator<<(unsigned int val);
  //! Does nothing.
  NullOutStream& operator<<(long val);
  //! Does nothing.
  NullOutStream& operator<<(unsigned long val);
  //! Does nothing.
  NullOutStream& operator<<(float val);
  //! Does nothing.
  NullOutStream& operator<<(double val);
  //! Does nothing.
  NullOutStream& operator<<(long double val);
  //! Does nothing.
  NullOutStream& operator<<(void* val);
  //! Does nothing.
  NullOutStream& operator<<(const char* str);
  //! Does nothing.
  NullOutStream& operator<<(std::string& str);
  //! Does nothing.
  NullOutStream& operator<<(std::streambuf* sb);
  //! Does nothing.
  NullOutStream& operator<<(std::ostream& (*pf) (std::ostream&));
  //! Does nothing.
  NullOutStream& operator<<(std::ios& (*pf) (std::ios&));
  //! Does nothing.
  NullOutStream& operator<<(std::ios_base& (*pf) (std::ios_base&));

  //! Does nothing.
  template<typename T>
  NullOutStream& operator<<(T s)
  { return *this; }
};

} // namespace io
} // namespace mlpack

#endif
