/**
 * @file log.hpp
 * @author Marcus Edel
 *
 * Definition of the Info and Warn log functions.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_LOG_HPP
#define ENSMALLEN_LOG_HPP

namespace ens {
/**
 * This class does nothing and should be optimized out entirely by the compiler.
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

#ifdef ENS_PRINT_INFO
  static std::ostream& Info = arma::get_cout_stream();
#else
  static NullOutStream Info;
#endif

#ifdef ENS_PRINT_WARN
  static std::ostream& Warn = arma::get_cerr_stream();
#else
  static NullOutStream Warn;
#endif

} // namespace ens

#endif
