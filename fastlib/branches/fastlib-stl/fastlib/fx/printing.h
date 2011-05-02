/***
 * @file printing.h
 * @author Ryan Curtin
 *
 * Four classes that define the output levels for IO.  They are very simply and
 * simply prepend an output level notifier to the output.
 */

#ifndef MLPACK_IO_PRINTING_H
#define MLPACK_IO_PRINTING_H

#include <iostream>
#include <map>

#define TYPENAME(x) (std::string(typeid(x).name()))

namespace mlpack {
namespace io {

/***
 * The PrefixedOutStream class allows us to output to an ostream with a prefix
 * at the beginning of each line, in the same way we would output to cout or
 * cerr.  The prefix is specified in the constructor (as well as the
 * destination).  A newline is automatically included at the end of each call to
 * operator<<.  So, for example,
 *
 * PrefixedOutStream outstr(std::cout, "[TEST] ");
 * outstr << "Hello world I like " << 7.5;
 *
 * would give, on std::cout,
 *
 * [TEST] Hello world I like 7.5
 *
 * These objects are used for the IO logging levels (DEBUG, INFO, WARN, and
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
  PrefixedOutStream(std::ostream& destination, const char* prefix) :
    destination(destination), prefix(prefix) { /* nothing to do */ }

  /***
   * This is meant to be used similarly to the std::cout and std::cerr ostreams.
   * Pass an arbitrary object in, and the prefix will be prepended, a newline
   * appended, and it will be sent to std::cout.
   */
  template<typename T>
  std::ostream& operator<<(const T& rhs) {
    destination << prefix << rhs;
    return destination;
  }

 private:
  std::ostream& destination;
  const char* prefix;
};

/***
 * The NullOutStream is used in place of the PrefixedOutStream for the IO debug
 * logging object when the program is compiled without debugging symbols.  It
 * does nothing, so the hope is that an optimizer will realize that it is doing
 * nothing and remove it.
 */
class NullOutStream {
 public:
  /***
   * This function intentionally does nothing.  It should be optimized out.
   */
  template<typename T>
  std::ostream& operator<<(const T& rhs) {
    return std::cout;
  }
};

class Printing {
  public:
   /**
   Prints the value of the given variable, by finding the class registered
  handle a particular type, specified by the integer ID given by
typeid(..).
  */
  static void PrintValue(std::string& id, std::string& pathname);
  protected:
   /* Maps a type to the appropriate printing class. */
   static std::map<std::string, Printing*> castingMap;
 
   /* Registers a printing class in the map. */
   Printing(std::string id);

   /* Prints the actual data */
   virtual void ToString(std::string& pathname)=0;
};
 
 
class IntPrinter : public Printing{
  IntPrinter();
  public:
   void ToString(std::string& pathname);
   static IntPrinter tmp;
};
 
class StringPrinter : public Printing{
 StringPrinter();
  public:
   void ToString(std::string& pathname);
   static StringPrinter tmp;
};

class TimerPrinter : public Printing{
  TimerPrinter();
 public:
  void ToString(std::string& pathname);
  static TimerPrinter tmp;
};

};
};
#endif 
