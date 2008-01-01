#include "ortho_range_search.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);
  bool do_naive = fx_param_exists(NULL, "do_naive");

  OrthoRangeSearch fast_search;

  fast_search.Init();
  fast_search.Compute();

  if(fx_param_exists(NULL, "save_tree_file")) {
    fast_search.SaveTree();
  }

  ArrayList<bool> fast_search_results = fast_search.get_results();

  // if naive option is specified, do naive algorithm
  if(do_naive) {
    NaiveOrthoRangeSearch search;
    search.Init();
    search.Compute();
    ArrayList<bool> naive_search_results = search.get_results();
    bool flag = true;

    for(index_t i = 0; i < fast_search_results.size(); i++) {

      if(fast_search_results[i] != naive_search_results[i]) {
	flag = false;
	printf("Differ on %d\n", i);
	break;
      }
    }
    if(flag) {
      printf("Both naive and tree-based method got the same results...\n");
    }
    else {
      printf("Both methods have different results...\n");
    }
  }
  timer *tree_range_search_time = fx_timer(NULL, "tree_range_search");
  timer *naive_search_time = fx_timer(NULL, "naive_search");
	 
  fx_format_result(NULL, "speedup", "%g", 
		   ((double)(naive_search_time->total).micros) / 
		   ((double)(tree_range_search_time->total).micros));
  fx_done();
  return 0;
}
