#include "ortho_range_search.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);
  bool do_naive = fx_param_exists(NULL, "do_naive");

  OrthoRangeSearch fast_search;
  fast_search.Init();
  fast_search.Compute();

  // if naive option is specified, do naive algorithm
  if(do_naive) {
    NaiveOrthoRangeSearch search; 
    search.Init();    
    search.Compute();
  }
  fx_done();
  return 0;
}
