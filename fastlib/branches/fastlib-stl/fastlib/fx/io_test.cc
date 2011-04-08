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
PARAM_TIMER(timer, "desc", allnn);

int main(int argc, char** argv) {
  IO::startTimer("allnn/timer");
  IO::parseCommandLine(argc, argv);
  IO::stopTimer("allnn/timer");
  IO::getValue<int>("testint") = 42;
  IO::print();
}

