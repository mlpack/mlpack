#include "fastlib/fastlib.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  Vector vec;
  int n = 4;
  vec.Init(n);
  for(int i = 0; i < n; i++) {
    vec[i] = i+1;
  }

  vec.PrintDebug("pre freeze vec");

  WriteOutOTObject("frozen_vector", vec);

  vec.PrintDebug("post freeze vec");

  
  //ReadInOTObject(

  
  
  fx_done(fx_root);
}
