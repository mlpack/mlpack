#ifndef PRINTING_H
#define PRINTING_H

//External includes
#include <boost/any.hpp>
#include <map>
#include <typeinfo>
#include <string>

#define TYPENAME(x) (std::string(typeid(x).name()))

namespace mlpack {
  class Printing {
    public:
      /** 
      Prints the value of the given variable, by finding the class registered to handle
      a particular type, specified by the integer ID given by typeid(..).
      */
      static void printValue(std::string& id, std::string& pathname);
    protected:
      static std::map<std::string, Printing*> castingMap;
      Printing(std::string id);
      virtual void toString(std::string& pathname)=0;
  };
  
  
  class IntPrinter : public Printing{
      IntPrinter();
    public:
      void toString(std::string& pathname);
      static IntPrinter tmp;
  };
  
    class StringPrinter : public Printing{
      StringPrinter();
    public:
      void toString(std::string& pathname);
      static StringPrinter tmp;
  };
  
    class TimerPrinter : public Printing{
      TimerPrinter();
    public:
      void toString(std::string& pathname);
      static TimerPrinter tmp;
  };
  
  class BoolPrinter : public Printing{
      BoolPrinter();
    public:
      void toString(std::string& pathname);
      static BoolPrinter tmp;
  };
};

#endif 