/**
 * @file fastica_main.cc
 *
 * Demonstrates usage of fastica.h
 *
 * @see fastica.h
 *
 * @author Nishant Mehta
 */
#include "fastica.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  srand48(time(0));

  Matrix X, W, Y;

  const char *data = fx_param_str_req(NULL, "data");
  data::Load(data, &X);

  struct datanode* fastica_module =
    fx_submodule(NULL, "fastica", "fastica_module");

  FastICA fastica;

  fastica.Init(X, fastica_module);

  int ret_val = fastica.DoFastICA(&W, &Y);

  SaveCorrectly("unmixing_matrix.dat", W);
  SaveCorrectly("indep_comps.dat", Y);

  if(ret_val == SUCCESS_PASS) {
    printf("PASSED");
  }
  else {
    printf("FAILED!");
  }

  //fx_done();

  return ret_val;
}
