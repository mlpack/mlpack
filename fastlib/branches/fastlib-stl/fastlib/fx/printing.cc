#include "printing.h"
#include "io.h"

#include <string>
#include <map>
#include <sys/time.h>

using namespace mlpack;

std::map<std::string, Printing*> Printing::castingMap = std::map<std::string, Printing*>();

Printing::Printing(std::string id) {
  castingMap.insert(std::pair<std::string, Printing*>(id, this));
}

void Printing::printValue(std::string& id, std::string& pathname) {
  //Is there a handler registered for this type?  Is that a valid pathname?
  if(!castingMap.count(id)) {
    //IO::printWarn("No handler registered");
    return;
  }
  if(!IO::checkValue(pathname.c_str())) {
    //IO::printWarn(pathname.c_str());
    return;
  }
  
  //Great! Lets print it.
  castingMap[id]->toString(pathname);
};

IntPrinter IntPrinter::tmp;
IntPrinter::IntPrinter() : Printing(TYPENAME(int)) {
};

void IntPrinter::toString(std::string& pathname) {
  std::cout << "d" << IO::getValue<int>(pathname.c_str());
};

//String printer
StringPrinter StringPrinter::tmp;
StringPrinter::StringPrinter() : Printing(TYPENAME(std::string)) {
};

void StringPrinter::toString(std::string& pathname) {
  std::cout << " " << IO::getValue<std::string>(pathname.c_str());
};

//Timer printer
TimerPrinter TimerPrinter::tmp;
TimerPrinter::TimerPrinter() : Printing(TYPENAME(timeval)) {
};

void TimerPrinter::toString(std::string& pathname) {
  std::cout << " sec: " << IO::getValue<timeval>(pathname.c_str()).tv_sec << 
                                              " usec: " << IO::getValue<timeval>(pathname.c_str()).tv_usec;
};

//Bool printer
BoolPrinter BoolPrinter::tmp;
BoolPrinter::BoolPrinter() : Printing(TYPENAME(bool)) {
};

void BoolPrinter::toString(std::string& pathname) {
  std::cout << " " << IO::getValue<bool>(pathname.c_str()) << std::endl;
};



