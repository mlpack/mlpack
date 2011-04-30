#include "io.h"
#include "optionshierarchy.h"
#include "printing.h"

#include <string>
#include <iostream>
#include <sys/time.h>
#include <typeinfo>

using namespace mlpack;


//PARAM_CUSTOM(int, "testint", "blarg");
PARAM_INT("testint", "cool", "");
//PARAM_CUSTOM(std::string, "str", "shiazer");
PARAM_INT("tmp", "cool, dude", "allnn");
PARAM_BOOL("perm", "not cool", "allknn");
//TIMER("timer", "desc", "allnn");

int main(int argc, char** argv) {
  IO::StartTimer("allnn/timer");
  IO::ParseCommandLine(argc, argv);
  IO::StopTimer("allnn/timer");
  IO::GetValue<int>("testint") = 42;

  // Some output
  IO::Debug << "A test of debugging output.";
  IO::Info << "A test of info output.";
  IO::Warn << "A test of warning output.";
  IO::Fatal << "A test of fatal output.";
}

