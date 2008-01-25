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

  Matrix X;
  const char* data = fx_param_str_req(NULL, "data");
  data::Load(data, &X);

  const char* ic_filename =
    fx_param_str(NULL, "ic_filename", "ic.dat");
  
  const char* unmixing_filename =
    fx_param_str(NULL, "unmixing_filename", "unmixing.dat");
  
  struct datanode* fastica_module =
    fx_submodule(NULL, "fastica", "fastica_module");

  FastICA fastica;

  int success_status = SUCCESS_FAIL;
  if(fastica.Init(X, fastica_module) == SUCCESS_PASS) {
    Matrix W, Y;
    if(fastica.DoFastICA(&W, &Y) == SUCCESS_PASS) {
      SaveCorrectly(unmixing_filename, W);
      //data::Save(ic_filename, Y);
      success_status = SUCCESS_PASS;
      VERBOSE_ONLY( W.PrintDebug("W") );
    }
  }
  

  if(success_status == SUCCESS_FAIL) {
    VERBOSE_ONLY( printf("FAILED!\n") );
  }

  fx_done();

  return success_status;
}
