#include "fastlib/fastlib_int.h"
#include "regression_new1.h"

int main(int argc, char *argv[]){

  fx_init(argc,argv);

  Regression_new1 ren1;
  ren1.Init();
  ren1.Compute();
}
