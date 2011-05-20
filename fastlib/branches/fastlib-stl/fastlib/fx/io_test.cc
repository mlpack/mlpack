#include "io.h"
#include "optionshierarchy.h"
#include "printing.h"

#include <string>
#include <iostream>
#include <sys/time.h>
#include <typeinfo>

#define DEFAULT_INT 42

using namespace mlpack;
PARAM_INT("gint", "global desc", "global");


bool TestIO();
bool TestHierarchy();
bool TestOption();

int main(int argc, char** argv) {
  IO::GetValue<int>("global/gint") = DEFAULT_INT;

  if (TestIO())
    IO::Info << "Test IO Succeeded." << std::endl;
  else
    IO::Fatal << "Test IO Failed." << std::endl;

  if (TestHierarchy())
    IO::Info << "Test Hierarchy Passed." << std::endl;
  else
    IO::Fatal << "Test Hierarchy Failed." << std::endl;

  if (TestOption())
    IO::Info << "Test Option Passed." << std::endl;
  else
    IO::Fatal << "Test Option Failed." << std::endl;

}



bool TestIO() {
  return false;
}

bool TestHierarchy() {
  return false;
}

bool TestOption() {
  //This test will involve creating an option, and making sure IO reflects this.
  PARAM(int, "test", "test desc", "test_parent", DEFAULT_INT, false);
  IO::GetValue<int>("test_parent/test");
  
  //Does IO reflect this?
  if (!IO::CheckValue("test_parent/test"))
    return false;
  IO::Debug << "CheckValue passed." << std::endl;

  if (IO::GetDescription("test_parent/test").compare(std::string("test desc")) != 0)
    return false;

  IO::Debug << "GetDescription passed." << std::endl;

  if (IO::GetValue<int>("test_parent/test") != DEFAULT_INT)
    return false;

  IO::Debug << "GetValuePassed." << std::endl;
  return true;
}
