#include "io.h"
#include "optionshierarchy.h"
#include <string>

#include <iostream>
#include <sys/time.h>

using namespace mlpack;


//IO_FLAG(testint, "blarg", foo);

int main(int argc, char** argv) {
  
  IO::parseCommandLine(argc, argv);
  
  std::string& x = IO::getValue<std::string>("testint");
  x = "2o4";
  IO::startTimer("brak/timer");


  for(int i = 0; i < 10; i++) { 
    std::cout << "herp " << i << std::endl;
  }
  std::cout << "propogate?" << std::endl;
  IO::stopTimer("brak/timer");
  std::cout << IO::getValue<timeval>("brak/timer").tv_usec << std::endl;
}
