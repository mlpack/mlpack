#include "ortho_range_search.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);
  bool do_naive = fx_param_exists(NULL, "do_naive");

  OrthoRangeSearch fast_search;
  ArrayList<int> fast_search_results;
  fast_search_results.Init(0);
  fast_search.Init();
  fast_search.Compute();
  fast_search.get_results(fast_search_results);

  // if naive option is specified, do naive algorithm
  if(do_naive) {
    NaiveOrthoRangeSearch search;
    ArrayList<int> naive_search_results;
    naive_search_results.Init(0);
    search.Init();    
    search.Compute();
    search.get_results(naive_search_results);

    printf("Fast method got %d candidates, slow method got %d candidates\n",
	   fast_search_results.size(), naive_search_results.size());
  }
  fx_done();
  return 0;
}
