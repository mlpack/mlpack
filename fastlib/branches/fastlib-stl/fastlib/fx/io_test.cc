#include "io.h"
#include "optionshierarchy.h"
#include "printing.h"

#include <string>
#include <iostream>
#include <sys/time.h>
#include <typeinfo>

using namespace mlpack;


PARAM_CUSTOM(int, testint, "blarg");
PARAM_CUSTOM(std::string, str, "shiazer");
PARAM_INT(tmp, "cool, dude", allnn);
PARAM_BOOL(perm, "not cool", allknn);

void test(); 

int main(int argc, char** argv) {
  IO::parseCommandLine(argc, argv);
  IO::getValue<int>("testint") = 42;
  test();
  IO::print();
}

void test() {
  std::cout << IO::getValue<int>("testint") ;
}