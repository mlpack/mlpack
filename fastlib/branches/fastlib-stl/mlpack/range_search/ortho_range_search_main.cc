/** @file ortho_range_search_main.cc
 *
 *  A test driver for the orthogonal range search code.
 *
 *  @author Dongryeol Lee (dongryel)
 */
#include "data_aux.h"
#include "naive_ortho_range_search.h"
#include "ortho_range_search.h"

#include "fastlib/mmanager/memory_manager.h"

#include <fastlib/fx/io.h>

void OutputOrhogonalRangeSearchResults(const GenMatrix<bool> &search_results,
				       const char *file_name) {
  
  FILE *file_pointer = fopen(file_name, "w+");

  for(index_t r = 0; r < search_results.n_rows(); r++) {
    for(index_t c = 0; c < search_results.n_cols(); c++) {

      if(search_results.get(r, c)) {
	fprintf(file_pointer, "1 ");
      }
      else {
	fprintf(file_pointer, "0 ");
      }
    }
    fprintf(file_pointer, "\n");
  }
}

/** Main function which reads parameters and runs the orthogonal
 *  range search.
 *
 *  In order to compile this driver, do:
 *  fl-build ortho_range_search_bin --mode=fast
 *
 *  Type the following (which consists of both required and optional
 *  arguments) in a single command line:
 *
 *  ./ortho_range_search_bin --data=dataset
 *                           --range=range
 *                           --do_naive
 *                           --save_tree_file=savedtree
 *                           --load_tree_file=savedtree
 *
 *  Explanations for the arguments listed with possible values.
 *
 *  1. data (required): the name of the dataset.
 *
 *  2. range (required): the textfile containing the range. The following
 *                       is an example:
 *
 *                       0.88 INFINITY
 *
 *                       0.90 INFINITY
 *
 *                       0.90 INFINITY
 *
 *                       0.90 INFINITY
 *
 *                       for 4-dimensional search window of
 *                       [0.88,inf]*[0.90,inf]*[0.90,inf]*[0.90,inf]
 *
 *  3. do_naive (optional): if present, will run the naive algorithm to
 *                          compare the search result against the naive search.
 *  4. save_tree_file (optional): this feature is in a beta stage and WILL NOT
 *                                work with the current FastLib release. It 
 *                                saves the constructed tree in the fast 
 *                                algorithm to a file, so that it can be loaded
 *                                later via --load_tree_file argument. If you 
 *                                want to use this, let me 
 *                                (dongryel@cc.gatech.edu) know.
 *  5. load_tree_file (optional): this feature is in a beta stage and WILL NOT
 *                                work with the current FastLib release. It
 *                                loads the saved tree from a file, so that
 *                                the algorithm does not have to construct
 *                                trees. If you want to use this feature, let
 *                                me (dongryel@cc.gatech.edu) know.
 */

PARAM_STRING_REQ("data", "The name of the dataset", "range");
PARAM_STRING_REQ("range", "Textfile containing the range", "range");

PARAM_STRING("lower", "Undocumented parameter", "range", "lower.csv");
PARAM_STRING("upper", "Undocumented parameter", "range", "upper.csv");
PARAM_STRING("load_tree_file", "Undocumented parameter", "range", "");
PARAM_STRING("save_tree_file", "Undocumented parameter", "range", "");

PARAM_FLAG("do_naive", "Indicates that the naive algorithm \
will be compared with the regular algorithm", "range");


using namespace mlpack;

int main(int argc, char *argv[]) {

  // Always initialize FASTexec with main's inputs at the beggining of
  // your program.  This reads the command line, among other things.
  IO::ParseCommandLine(argc, argv);

  // Initialize Nick's memory manager...
  std::string pool_name("/scratch/temp_mem");
  mmapmm::MemoryManager<false>::allocator_ =
    new mmapmm::MemoryManager<false>();
  mmapmm::MemoryManager<false>::allocator_->set_pool_name(pool_name);
  mmapmm::MemoryManager<false>::allocator_->set_capacity(65536*1000);
  mmapmm::MemoryManager<false>::allocator_->Init();

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // The reference data file is a required parameter.
  const char* dataset_file_name = IO::GetParam<std::string>("range/data");

  // column-oriented dataset matrix.
  GenMatrix<double> dataset;

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(dataset_file_name, &dataset);

  // Generate the 200 uniformly distributed search ranges.
  GenMatrix<double> low_coord_limits, high_coord_limits;
  
  // The file containing the lower and the upper limits of each search
  // window.
  const char *lower_limit_file_name = IO::GetParam<std::string>("range/lower").c_str();
  const char *upper_limit_file_name = IO::GetParam<std::string>("range/upper").c_str();

  data::Load(lower_limit_file_name, &low_coord_limits);
  data::Load(upper_limit_file_name, &high_coord_limits);

  // flag for determining whether we need to do naive algorithm.
  bool do_naive = IO::HasParam("range/do_naive");

  // File name containing the saved tree (if the user desires to do so)
  const char *load_tree_file_name;

  if(IO::HasParam("range/load_tree_file")) {
    load_tree_file_name = IO::GetParam<std::string>("range/load_tree_file").c_str();
  }
  else {
    load_tree_file_name = NULL;
  }

  // Declare fast tree-based orthogonal range search algorithm object.
  OrthoRangeSearch<double> fast_search;
  fast_search.Init(dataset, IO::HasParam("range/do_naive"),
		   load_tree_file_name);
  GenMatrix<bool> fast_search_results;
  fast_search.Compute(low_coord_limits, high_coord_limits, 
		      &fast_search_results);

  if(IO::HasParam("range/save_tree_file")) {
    const char *save_tree_file_name = IO::GetParam<std::string>("range/save_tree_file").c_str();
    fast_search.SaveTree(save_tree_file_name);
  }

  // Dump the result to the file.
  OutputOrhogonalRangeSearchResults(fast_search_results, "results.txt");

  // if naive option is specified, do naive algorithm
  if(do_naive) {
    NaiveOrthoRangeSearch<double> search;
    GenMatrix<bool> naive_search_results;
    search.Init(dataset);
    search.Compute(low_coord_limits, high_coord_limits, &naive_search_results);
    bool flag = true;

    for(index_t i = 0; i < fast_search_results.n_cols(); i++) {
      for(index_t j = 0; j < dataset.n_cols(); j++) {	
	if(fast_search_results.get(j, i) != naive_search_results.get(j, i)) {
	  flag = false;
	  IO::Info << "Differ on (" << i << ", " << j << ")" << std::endl;
	  break;
	}
      }
    }
    if(flag) {
      IO::Info << "Both naive and tree-based method got the same results..." << std::endl;
    }
    else {
      IO::Info << "Both methods have different results..." << std::endl;
    }
  }
	 
  IO::GetParam<double>("range/speedup") =
		   ((double)(IO::GetParam<timeval>("range/naive_search").tv_usec) / 
		   (double)(IO::GetParam<timeval>("range/tree_range_search").tv_usec));

  // Destroy Nick's memory manager...
  mmapmm::MemoryManager<false>::allocator_->Destruct();
  delete mmapmm::MemoryManager<false>::allocator_;

  return 0;
}
