/***
 * @file printing.h
 * @author Ryan Curtin
 *
 * Allows the printing of several common primitives in the IO hierarchy.
 */

#ifndef MLPACK_IO_PRINTING_H
#define MLPACK_IO_PRINTING_H

#include <map>
#include <string>

#define TYPENAME(x) (std::string(typeid(x).name()))

namespace mlpack {
namespace io {

class Printing {
  public:
   /*
   * Prints the value of the given variable, by finding the class registered
   * to handle a particular type, specified by the integer ID given by
   * typeid(..).
   *
   * @param id ID of the type of the value.
   * @param pathname Pathname of the value to be printed. 
   */
   static void PrintValue(std::string& id, std::string& pathname);
   protected:
   
   /* Maps a type to the appropriate printing class. */
   static std::map<std::string, Printing*> castingMap;
 
   /* 
   * Registers a printing class in the map. 
   *
   * @param id Type to register the new instance to.
   */
   Printing(std::string id);

   /* 
   * Prints the actual data 
   * 
   * @param pathname Full pathname of the value to be printed. 
   */
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
