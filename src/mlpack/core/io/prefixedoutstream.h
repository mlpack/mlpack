#ifndef MLPACK_CLI_PREFIXED_OUT_STREAM_H
#define MLPACK_CLI_PREFIXED_OUT_STREAM_H

#include <iostream>
#include <iomanip>
#include <string>
#include <streambuf>
#include <boost/lexical_cast.hpp>
#include <armadillo>

namespace mlpack {
namespace io {
  
/***
 * @brief The PrefixedOutStream class allows us to output to an ostream with a 
 * prefix at the beginning of each line, in the same way we would output to 
 * cout or cerr.  The prefix is specified in the constructor (as well as the
 * destination).  A newline is automatically included at the end of each call 
 * to operator<<.  So, for example,
 *
 * PrefixedOutStream outstr(std::cout, "[TEST] ");
 * outstr << "Hello world I like " << 7.5;
 * outstr << "...Continue" << std::endl;
 * outstr << "After the CR" << std::endl;
 *
 * would give, on std::cout,
 *
 * [TEST] Hello world I like 7.5...Continue
 * [TEST] After the CR
 *
 * These objects are used for the CLI logging levels (DEBUG, INFO, WARN, and
 * FATAL).
 */
class PrefixedOutStream {
 public:
  /***
   * Set up the PrefixedOutStream.
   *
   * @param destination ostream which receives output from this object.
   * @param prefix The prefix to prepend to each line.
   */
  PrefixedOutStream(std::ostream& destination, const char* prefix,
      bool ignoreInput = false, bool fatal = false) :
      destination(destination), ignoreInput(ignoreInput), 
      prefix(prefix), carriageReturned(true), fatal(fatal)
      // We want the first call to operator<< to prefix the prefix so we set
      // carriageReturned to true.
    { /* nothing to do */ } 
 
  /**
   * @brief Each of these functions outputs the specified value.  If a newline 
   *   was encountered during the previous call, a prefix will be prefixed to
   *   the output.
   *
   * @param val The data to be output.
   *
   * @return Reference to this object, so as to allow chained '<<' operations.
   */
  PrefixedOutStream& operator<<(bool val);
  PrefixedOutStream& operator<<(short val);
  PrefixedOutStream& operator<<(unsigned short val);
  PrefixedOutStream& operator<<(int val);
  PrefixedOutStream& operator<<(unsigned int val);
  PrefixedOutStream& operator<<(long val);
  PrefixedOutStream& operator<<(unsigned long val);
  PrefixedOutStream& operator<<(float val);
  PrefixedOutStream& operator<<(double val);
  PrefixedOutStream& operator<<(long double val);
  PrefixedOutStream& operator<<(void* val);
  PrefixedOutStream& operator<<(const char* str);
  PrefixedOutStream& operator<<(std::string& str);
  PrefixedOutStream& operator<<(std::streambuf* sb);
  PrefixedOutStream& operator<<(std::ostream& (*pf)(std::ostream&));
  PrefixedOutStream& operator<<(std::ios& (*pf)(std::ios&));
  PrefixedOutStream& operator<<(std::ios_base& (*pf)(std::ios_base&));

  template<typename T>
  PrefixedOutStream& operator<<(T s);

  /**
   * @brief The output stream that all data is to be sent too.  Eg, cout.
   */
  std::ostream& destination;

  /**
   * @brief Discards input, prints nothing if true.
   */
  bool ignoreInput;

 private:
  /**
   * @brief Conducts the base logic required in all the operator << overloads.
   *   Mostly just a good idea to reduce copy-pasta.
   *
   * @tparam T The type of the data to output.
   * @param val The The data to be output.
   */
  template<typename T>
  void BaseLogic(T val);

  /***
   * Output the prefix, but only if we need to and if we are allowed to.
   */
  inline void PrefixIfNeeded();

  /**
   * @brief Contains a string which will be prefixed to the output after every 
   *   carriage return.
   */
  std::string prefix;

  /**
   * @brief If true, then the previous call to operator<< encountered a CR.  
   *   The prefix should therefore be prefixed.
   */
  bool carriageReturned;

  /**
   * @brief If set to true, the application will terminate with an error when a
   *    carriage return is encountered.
   */
  bool fatal;

};

// Template definitions
#include "prefixedoutstream_impl.h"

} // namespace io
} // namespace mlpack

#endif 
