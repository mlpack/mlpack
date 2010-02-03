/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
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
int main(int argc, char *argv[]) {

  // Always initialize FASTexec with main's inputs at the beggining of
  // your program.  This reads the command line, among other things.
  fx_init(argc, argv, NULL);

  // Initialize Nick's memory manager...
  std::string pool_name("/scratch/temp_mem");
  mmapmm::MemoryManager<false>::allocator_ =
    new mmapmm::MemoryManager<false>();
  mmapmm::MemoryManager<false>::allocator_->set_pool_name(pool_name);
  mmapmm::MemoryManager<false>::allocator_->set_capacity(65536*1000);
  mmapmm::MemoryManager<false>::allocator_->Init();

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // The reference data file is a required parameter.
  const char* dataset_file_name = fx_param_str(NULL, "data", "data.csv");

  // column-oriented dataset matrix.
  GenMatrix<double> dataset;

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(dataset_file_name, &dataset);

  // Generate the 200 uniformly distributed search ranges.
  GenMatrix<double> low_coord_limits, high_coord_limits;
  
  // The file containing the lower and the upper limits of each search
  // window.
  const char *lower_limit_file_name = fx_param_str(NULL, "lower", "lower.csv");
  const char *upper_limit_file_name = fx_param_str(NULL, "upper", "upper.csv");

  data::Load(lower_limit_file_name, &low_coord_limits);
  data::Load(upper_limit_file_name, &high_coord_limits);

  // flag for determining whether we need to do naive algorithm.
  bool do_naive = fx_param_exists(NULL, "do_naive");

  // File name containing the saved tree (if the user desires to do so)
  const char *load_tree_file_name;

  if(fx_param_exists(NULL, "load_tree_file")) {
    load_tree_file_name = fx_param_str(NULL, "load_tree_file", NULL);
  }
  else {
    load_tree_file_name = NULL;
  }

  // Declare fast tree-based orthogonal range search algorithm object.
  OrthoRangeSearch<double> fast_search;
  fast_search.Init(dataset, fx_param_exists(NULL, "do_naive"),
		   load_tree_file_name);
  GenMatrix<bool> fast_search_results;
  fast_search.Compute(low_coord_limits, high_coord_limits, 
		      &fast_search_results);

  if(fx_param_exists(NULL, "save_tree_file")) {
    const char *save_tree_file_name = fx_param_str(NULL, "save_tree_file",
						   NULL);
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
	  printf("Differ on (%d, %d)\n", i, j);
	  break;
	}
      }
    }
    if(flag) {
      printf("Both naive and tree-based method got the same results...\n");
    }
    else {
      printf("Both methods have different results...\n");
    }
  }
  fx_timer *tree_range_search_time = fx_get_timer(NULL, "tree_range_search");
  fx_timer *naive_search_time = fx_get_timer(NULL, "naive_search");
	 
  fx_format_result(NULL, "speedup", "%g", 
		   ((double)(naive_search_time->total).micros) / 
		   ((double)(tree_range_search_time->total).micros));

  // Destroy Nick's memory manager...
  mmapmm::MemoryManager<false>::allocator_->Destruct();
  delete mmapmm::MemoryManager<false>::allocator_;
  fx_done(NULL);
  return 0;
}
