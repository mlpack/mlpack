/**
 * @file main.cc
 *
 * Test driver for our infomax ICA method.
 */

#include "infomax_ica.h"
#include "test_infomax_ica.h"
#include "fastlib/fastlib.h"
#include "data/dataset.h"
int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  TestInfomaxICA *testica = new TestInfomaxICA();
  testica->Init();
  testica->TestAll();

  fx_done();
}
