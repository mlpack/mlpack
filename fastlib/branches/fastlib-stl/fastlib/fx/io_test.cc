#include "io.h"
#include "optionshierarchy.h"
#include <string>

#include <iostream>
#include <sys/time.h>

using namespace mlpack;

int main(int argc, char** argv) {
  IO::add("test", "description"); 
  IO::add("help", "This is a help message", "/bar/foo/");
  IO::add("foo", "durka durka", "bar");
  IO::add("help1", "This is a help message", "/bar/foo");
  IO::add("help2", "This is a help message", "barfoo");
  IO::add("help3", "This is a help message", "/bar/foo/");
  IO::add("bar", "This is a submodule");
  //IO::add<std::string>("test3",  "This should be a string", NULL, true); 
  IO::add<std::string>("testint", "This should be an int", NULL, true);
  
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
