#include "printing.h"
#include "io.h"

#include <string>
#include <map>
#include <sys/time.h>

using namespace mlpack::io;


std::map<std::string, Printing*> Printing::castingMap = 
  std::map<std::string, Printing*>();

Printing::Printing(std::string id) {
  castingMap.insert(std::pair<std::string, Printing*>(id, this));
}

Printing::~Printing() { }

void Printing::PrintValue(std::string& id, std::string& pathname) {
  //Is there a handler registered for this type?  Is that a valid pathname?
  if (!id.length() || !castingMap.count(id)) {
    return;
  }
  //Great! Lets print it.
  if (castingMap[id] != NULL) {    
    castingMap[id]->ToString(pathname);
  }
};

IntPrinter IntPrinter::tmp;
IntPrinter::IntPrinter() : Printing(TYPENAME(int)) {
};
IntPrinter::~IntPrinter() { }

void IntPrinter::ToString(std::string& pathname) {
  if (IO::CheckValue(pathname.c_str()))
    std::cout << IO::GetValue<int>(pathname.c_str());
};

//String printer
StringPrinter StringPrinter::tmp;
StringPrinter::StringPrinter() : Printing(TYPENAME(std::string)) {
};
StringPrinter::~StringPrinter() { }

void StringPrinter::ToString(std::string& pathname) {
  if (IO::CheckValue(pathname.c_str()))
    std::cout << IO::GetValue<std::string>(pathname.c_str());
};

//Timer printer
TimerPrinter TimerPrinter::tmp;
TimerPrinter::TimerPrinter() : Printing(TYPENAME(timeval)) {
};
TimerPrinter::~TimerPrinter() { }

void TimerPrinter::ToString(std::string& pathname) {
  if (IO::CheckValue(pathname.c_str()))
    std::cout << " sec: " << IO::GetValue<timeval>(pathname.c_str()).tv_sec <<
      " usec: " << IO::GetValue<timeval>(pathname.c_str()).tv_usec;
};




