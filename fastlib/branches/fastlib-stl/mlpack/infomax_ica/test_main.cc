/**
 * @file main.cc
 *
 * Test driver for our infomax ICA method.
 */

#include <fastlib/fx/io.h>
#include "infomax_ica.h"
#include "test_infomax_ica.h"
#include "fastlib/fastlib.h"


using namespace mlpack;

int main(int argc, char *argv[]) {
  IO::ParseCommandLine(argc, argv);

  TestInfomaxICA *testica = new TestInfomaxICA();
  testica->Init();
  testica->TestAll();
}
