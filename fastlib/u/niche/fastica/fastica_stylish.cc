/**
 * @file fastica_stylish.cc
 *
 * Demonstrates usage of fastica_stylish.h
 *
 * @see fastica_stylish.h
 *
 * @author Nishant Mehta
 */
#include "fastica_stylish.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  srand48(time(0));

  Matrix X, W, Y;

  const char *data = fx_param_str_req(NULL, "data");
  data::Load(data, &X);

  FastICA fastica;

  fastica.Init(X);

  int ret_val = fastica.DoFastICA(fx_root, &W, &Y);

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
