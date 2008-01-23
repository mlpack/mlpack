/** @file ortho_range_search_main.cc
 *
 *  A test driver for the orthogonal range search code.
 *
 *  @author Dongryeol Lee (dongryel)
 */
#include "ortho_range_search.h"

int main(int argc, char *argv[]) {

  // Always initialize FASTexec with main's inputs at the beggining of
  // your program.  This reads the command line, among other things.
  fx_init(argc, argv);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // The reference data file is a required parameter.
  const char* dataset_file_name = fx_param_str_req(NULL, "data");

  // column-oriented dataset matrix.
  Matrix dataset;

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(dataset_file_name, &dataset);

  // flag for determining whether we need to do naive algorithm.
  bool do_naive = fx_param_exists(NULL, "do_naive");

  // declare fast tree-based orthogonal range search algorithm object.
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
    search.Init(dataset);
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
