/*
 * =====================================================================================
 *
 *       Filename:  main.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/18/2007 06:21:53 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include "fastlib/fastlib.h"
#include "u/nvasil/mmanager/memory_manager.h"
#include "u/nvasil/mmanager_with_tpie/memory_manager.h"
#include "u/nvasil/dataset/binary_dataset.h"
#include "u/nvasil/tree/tree_definitions.h"

struct Parameters {
  int32 dimension_;
	int64 num_of_points_;
	std::string data_file_;
	std::string out_file_;
	std::string temp_dir_;
	std::string memory_engine_;
	int32 page_size_;
	int32 knns_;
	std::string memory_file_;
  BinaryDataset<float32> data_;	
	uint64 capacity_;
	
};

template<typename TREE>
void DuallTreeAllNearestNeighbors(Parameters &args);

int main(int argc, char *argv[]) {
  MM_manager.ignore_memory_limit();
	Parameters args;
	// initialize command line parameter
	fx_init(argc, argv);
	args.temp_dir_=fx_param_str(NULL, "temp_dir", "./");
  if (fx_param_exists(NULL, "data_file")==1) {
	  args.data_file_=fx_param_str(NULL, "data_file", NULL);
		args.data_.Init(args.data_file_);
		args.num_of_points_= args.data_.get_num_of_points();
		args.dimension_= args.data_.get_dimension();
	} else {
		// if data file is not specified use random points
		printf("Generating random points...\n");
	  args.data_file_= args.temp_dir_+string("temp");
		args.dimension_= fx_param_int_req(NULL, "dimension");
		args.num_of_points_= fx_param_int_req(NULL, "num_of_points");
		args.data_.Init(args.data_file_, args.num_of_points_, args.dimension_);
		for(int64 i=0; i<args.num_of_points_; i++) {
		  for(int32 j=0; j<args.dimension_; j++) {
			  args.data_.At(i,j) = 1.0*rand()/RAND_MAX;
				args.data_.set_id(i,i);
			}
		}
	}	
	args.out_file_ = fx_param_str(NULL, "out_file", "allnn");
  args.capacity_ = fx_param_int(NULL, "capacity", 134217728);
	args.knns_ = fx_param_int(NULL, "knns", 2);
	args.page_size_=fx_param_int(NULL, "page_size", 4096);
  args.memory_file_ = fx_param_str(NULL, "memory_file", "temp_mem"); 
  args.memory_engine_ = fx_param_str(NULL, "memory_engine", "mmapmm");
	printf("Creating swap file...\n");
 	if (args.memory_engine_ == "mmapmm") {
		mmapmm::MemoryManager<false>::allocator_ = 
		    new mmapmm::MemoryManager<false>();
    mmapmm::MemoryManager<false>::
			  allocator_->set_capacity(args.capacity_);

		mmapmm::MemoryManager<false>::
			  allocator_->set_pool_name(args.memory_file_);

		mmapmm::MemoryManager<false>::allocator_->Init();
		DuallTreeAllNearestNeighbors<BinaryKdTreeMMAPMM_t>(args);    	  
    if (!fx_param_exists(NULL, "data_file")) {
		  unlink(args.data_file_.c_str()); 
		}	
	} else {
	  if (args.memory_engine_ == "tpiemm"){
      tpiemm::MemoryManager<false>::allocator_ = 
		      new tpiemm::MemoryManager<false>();
      tpiemm::MemoryManager<false>::
			    allocator_->set_cache_size(args.capacity_);

		  tpiemm::MemoryManager<false>::
			    allocator_->set_page_size(args.page_size_);
      
		  tpiemm::MemoryManager<false>::
			  allocator_->set_cache_file(args.memory_file_);

		  tpiemm::MemoryManager<false>::allocator_->Init();

	    DuallTreeAllNearestNeighbors<BinaryKdTreeTPIEMM_t>(args);    	  
      if (!fx_param_exists(NULL, "data_file")) {
		    unlink(args.data_file_.c_str()); 
		  }	
		} else {
		  FATAL("%s for memory_engine is not a valid option\n", 
					  args.memory_engine_.c_str());
		}
	}
	fx_format_result(NULL, "success", "%d", 1);
	fx_done();
  unlink(args.memory_file_.c_str());
}

template<typename TREE>
void DuallTreeAllNearestNeighbors(Parameters &args) {
  TREE tree;
	printf("Building the tree...");
	tree.Init(&args.data_);

	fx_timer_start(NULL, "build");	
	tree.BuildDepthFirst();
	fx_timer_stop(NULL, "build");
  printf("Memory usage: %llu\n",
	        (unsigned long long)TREE::Allocator_t::allocator_->get_usage());
	printf("%s\n", tree.Statistics().c_str());
	args.data_.Destruct();
	printf("Initializing all nearest neighbor output...\n");
  tree.InitAllKNearestNeighborOutput(args.out_file_, 
				                              args.knns_);
	printf("Computing all nearest neighbors...\n");
	fflush(stdout);
  fx_timer_start(NULL, "dualltree");	
	tree.AllNearestNeighbors(tree.get_parent(), args.knns_);
	fx_timer_stop(NULL, "dualltree");
	tree.CloseAllKNearestNeighborOutput(args.knns_);
  unlink(args.out_file_.c_str());	
}

