/**
 * @file core/util/prefixedoutstream.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Declaration of the PrefixedOutStream class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_PREFIXEDOUTSTREAM_HPP
#define MLPACK_CORE_UTIL_PREFIXEDOUTSTREAM_HPP

#include <mlpack/base.hpp>

namespace mlpack {
namespace util {

/**
 * Allows us to output to an ostream with a prefix at the beginning of each
 * line, in the same way we would output to cout or cerr.  The prefix is
 * specified in the constructor (as well as the destination ostream).  A newline
 * must be passed to the stream, and then the prefix will be prepended to the
 * next line.  For example,
 *
 * @code
 * PrefixedOutStream outstr(MLPACK_COUT_STREAM, "[TEST] ");
 * outstr << "Hello world I like " << 7.5;
 * outstr << "...Continue" << std::endl;
 * outstr << "After the CR\n" << std::endl;
 * @endcode
 *
 * would give, on MLPACK_COUT_STREAM,
 *
 * @code
 * [TEST] Hello world I like 7.5...Continue
 * [TEST] After the CR
 * [TEST]
 * @endcode
 *
 * These objects are used for the mlpack::Log levels (DEBUG, INFO, WARN, and
 * FATAL).
 */
class PrefixedOutStream
{
 public:
  /**
   * Set up the PrefixedOutStream.
   *
   * @param destination ostream which receives output from this object.
   * @param prefix The prefix to prepend to each line.
   * @param ignoreInput If true, the stream will not be printed.
   * @param fatal If true, a std::runtime_error exception is thrown after
   *     printing a newline.
   * @param backtrace If true, attempt to print a backtrace (will only be
   *     done if MLPACK_HAS_BFD_DL is defined).
   */
  PrefixedOutStream(std::ostream& destination,
                    const char* prefix,
                    bool ignoreInput = false,
                    bool fatal = false,
                    bool backtrace = true) :
      destination(destination),
      ignoreInput(ignoreInput),
      backtrace(backtrace),
      prefix(prefix),
      // We want the first call to operator<< to prefix the prefix so we set
      // carriageReturned to true.
      carriageReturned(true),
      fatal(fatal)
    { /* nothing to do */ }

  //! Write a bool to the stream.
  PrefixedOutStream& operator<<(bool val);
  //! Write a short to the stream.
  PrefixedOutStream& operator<<(short val);
  //! Write an unsigned short to the stream.
  PrefixedOutStream& operator<<(unsigned short val);
  //! Write an int to the stream.
  PrefixedOutStream& operator<<(int val);
  //! Write an unsigned int to the stream.
  PrefixedOutStream& operator<<(unsigned int val);
  //! Write a long to the stream.
  PrefixedOutStream& operator<<(long val);
  //! Write an unsigned long to the stream.
  PrefixedOutStream& operator<<(unsigned long val);
  //! Write a float to the stream.
  PrefixedOutStream& operator<<(float val);
  //! Write a double to the stream.
  PrefixedOutStream& operator<<(double val);
  //! Write a long double to the stream.
  PrefixedOutStream& operator<<(long double val);
  //! Write a void pointer to the stream.
  PrefixedOutStream& operator<<(void* val);
  //! Write a character array to the stream.
  PrefixedOutStream& operator<<(const char* str);
  //! Write a string to the stream.
  PrefixedOutStream& operator<<(std::string& str);
  //! Write a streambuf to the stream.
  PrefixedOutStream& operator<<(std::streambuf* sb);
  //! Write an ostream manipulator function to the stream.
  PrefixedOutStream& operator<<(std::ostream& (*pf)(std::ostream&));
  //! Write an ios manipulator function to the stream.
  PrefixedOutStream& operator<<(std::ios& (*pf)(std::ios&));
  //! Write an ios_base manipulator function to the stream.
  PrefixedOutStream& operator<<(std::ios_base& (*pf)(std::ios_base&));

  //! Write anything else to the stream.
  template<typename T>
  PrefixedOutStream& operator<<(const T& s);

  //! The output stream that all data is to be sent to; example:
  //! MLPACK_COUT_STREAM.
  std::ostream& destination;

  //! Discards input, prints nothing if true.
  bool ignoreInput;

  //! If true, on a fatal error, a backtrace will be printed if
  //! MLPACK_HAS_BFD_DL is ! defined.
  bool backtrace;

 private:
  /**
   * Conducts the base logic required in all the operator << overloads.  Mostly
   * just a good idea to reduce copy-paste.
   *
   * This overload is for non-Armadillo objects, which need special handling
   * during printing.
   *
   * @tparam T The type of the data to output.
   * @param val The The data to be output.
   */
  template<typename T>
  std::enable_if_t<!arma::is_arma_type<T>::value>
  BaseLogic(const T& val);

  /**
   * Conducts the base logic required in all the operator << overloads.  Mostly
   * just a good idea to reduce copy-paste.
   *
   * This overload is for Armadillo objects, which need special handling during
   * printing.
   *
   * @tparam T The type of the data to output.
   * @param val The The data to be output.
   */
  template<typename T>
  std::enable_if_t<arma::is_arma_type<T>::value>
  BaseLogic(const T& val);

  /**
   * Output the prefix, but only if we need to and if we are allowed to.
   */
  inline void PrefixIfNeeded();

  //! Contains the prefix we must prepend to each line.
  std::string prefix;

  //! If true, the previous call to operator<< encountered a CR, and a prefix
  //! will be necessary.
  bool carriageReturned;

  //! If true, a std::runtime_error exception will be thrown when a CR is
  //! encountered.
  bool fatal;
};

} // namespace util
} // namespace mlpack

// Template definitions.
#include "prefixedoutstream_impl.hpp"

#endif
