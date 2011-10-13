#ifndef MLPACK_CLI_NULL_OUT_STREAM_H
#define MLPACK_CLI_NULL_OUT_STREAM_H

#include <iostream>
#include <streambuf>
#include <string>

namespace mlpack {
namespace io {

/***
 * The NullOutStream is used in place of regular PrefixOutStreams for
 * the CLI debug output when DEBUG symbols are not defined.  It does nothing
 * Hopefully the optimizer will realize this and optimize it out.
 */
class NullOutStream {
 public:
  /**
   * @brief Does nothing.  Just like everything else in this nothing class.
   */
  NullOutStream();  
  NullOutStream(const NullOutStream& other);

  /**
   * @brief None of these functions do anything. 
   *
   * @param val Value to have nothing done with it.
   *
   * @return Reference to do nothing in the future.
   */
  NullOutStream& operator<<(bool val);
  NullOutStream& operator<<(short val);
  NullOutStream& operator<<(unsigned short val);
  NullOutStream& operator<<(int val);
  NullOutStream& operator<<(unsigned int val);
  NullOutStream& operator<<(long val);
  NullOutStream& operator<<(unsigned long val);
  NullOutStream& operator<<(float val);
  NullOutStream& operator<<(double val);
  NullOutStream& operator<<(long double val);
  NullOutStream& operator<<(void* val);
  NullOutStream& operator<<(const char* str);
  NullOutStream& operator<<(std::string& str);
  NullOutStream& operator<<(std::streambuf* sb);
  NullOutStream& operator<<(std::ostream& (*pf) (std::ostream&));
  NullOutStream& operator<<(std::ios& (*pf) (std::ios&));
  NullOutStream& operator<<(std::ios_base& (*pf) (std::ios_base&));

  template<typename T>
  NullOutStream& operator<<(T s) {
    return *this;
  }
 };

} // namespace io
} // namespace mlpack

#endif //MLPACK_CLI_NULL_OUT_STREAM_H
