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
  bool specialized_for_knns_;
  bool generate_points_only_;	
};

template<typename TREE>
void DuallTreeAllNearestNeighbors(Parameters &args);
template<typename TREE>
void DuallTreeAllNearestNeighborsSpecializedForKnn(Parameters &args);

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
	if (fx_param_exists(NULL, "gen_only")==1) {
	  NONFATAL("Generated random points only, no tree testing...\n");
		return 1;
	}
	args.out_file_ = fx_param_str(fx_root, "out_file", "allnn");
  args.capacity_ = fx_param_int(fx_root, "capacity", 134217728);
	args.knns_ = fx_param_int(fx_root, "knns", 2);
	args.page_size_ = fx_param_int(fx_root, "page_size", 4096);
  args.memory_file_ = fx_param_str(fx_root, "memory_file", "temp_mem"); 
  args.memory_engine_ = fx_param_str(fx_root, "memory_engine", "mmapmm");
	args.specialized_for_knns_ = fx_param_bool(fx_root, "specialized_for_knns", false);
	
	if (args.memory_engine_=="tpiemm") {
	 	unlink(args.memory_file_.c_str());
	}

 	if (sizeof(index_t)==sizeof(int32)) {
	  NONFATAL("index_t is int32, good for small scale problems");
	} else {
		if (sizeof(index_t)==sizeof(int64)) {
	    NONFATAL("index_t is int64, good for large scale problems");
		}	 
	}

	printf("Creating swap file...\n");
	if (args.memory_engine_ == "mmapmm") {
		mmapmm::MemoryManager<false>::allocator_ = 
		    new mmapmm::MemoryManager<false>();
    mmapmm::MemoryManager<false>::
			  allocator_->set_capacity(args.capacity_);

		mmapmm::MemoryManager<false>::
			  allocator_->set_pool_name(args.memory_file_);

		mmapmm::MemoryManager<false>::allocator_->Init();
		if (args.specialized_for_knns_==true) {
      DuallTreeAllNearestNeighborsSpecializedForKnn<
			    BinaryKdTreeMMAPMMKnnNode_t>(args);
		} else {
		  DuallTreeAllNearestNeighbors<BinaryKdTreeMMAPMM_t>(args);    	  
		}  
    if (!fx_param_exists(fx_root, "data_file")) {
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
     if (args.specialized_for_knns_==true) {
       DuallTreeAllNearestNeighborsSpecializedForKnn<
				 BinaryKdTreeTPIEMMKnnNode_t>(args);    	  
		 } else {
	     DuallTreeAllNearestNeighbors<BinaryKdTreeTPIEMM_t>(args);    	  
		 }
      if (!fx_param_exists(fx_root, "data_file")) {
		    unlink(args.data_file_.c_str()); 
		  }	
		} else {
		  FATAL("%s for memory_engine is not a valid option\n", 
					  args.memory_engine_.c_str());
		}
	}
	fx_format_result(fx_root, "success", "%d", 1);
	fx_done();
	if (args.memory_engine_=="tpiemm") {
	 	unlink(args.memory_file_.c_str());
	}

}

template<typename TREE>
void DuallTreeAllNearestNeighbors(Parameters &args) {
  TREE tree;
	printf("Building the tree...");
	fflush(stdout);
	tree.Init(&args.data_);

	fx_timer_start(fx_root, "build");
	tree.BuildDepthFirst();
	fx_timer_stop(fx_root, "build");
  NONFATAL("Memory usage: %llu\n",
	        (unsigned long long)TREE::Allocator_t::allocator_->get_usage());
	NONFATAL("%s\n", tree.Statistics().c_str());
	fflush(stdout);
	args.data_.Destruct();
	printf("Initializing all nearest neighbor output...\n");
  fx_timer_start(fx_root, "init_knn");
	tree.InitAllKNearestNeighborOutput(args.out_file_, 
				                              args.knns_);
	fx_timer_stop(fx_root, "init_knn");
	printf("Computing all nearest neighbors...\n");
	fflush(stdout);
  fx_timer_start(fx_root, "dualltree");	
	tree.AllNearestNeighbors(tree.get_parent(), args.knns_);
	fx_timer_stop(fx_root, "dualltree");
	tree.CloseAllKNearestNeighborOutput(args.knns_);
  unlink(args.out_file_.c_str());	
}

template<typename TREE>
void DuallTreeAllNearestNeighborsSpecializedForKnn(Parameters &args) {
  TREE tree;
	printf("Procceding with the specialized method for knn node..\n");
	tree.Init(&args.data_);
	tree.set_knns(args.knns_);
	fx_timer_start(fx_root, "build");	
  printf("Building the tree...\n");
	fflush(stdout);
	tree.BuildDepthFirst();
	fx_timer_stop(fx_root, "build");
  NONFATAL("Memory usage: %llu\n",
	        (unsigned long long)TREE::Allocator_t::allocator_->get_usage());
	NONFATAL("%s\n", tree.Statistics().c_str());
	args.data_.Destruct();
	printf("Computing all nearest neighbors...\n");
	fflush(stdout);
  fx_timer_start(fx_root, "dualltree");	
	tree.AllNearestNeighbors(tree.get_parent(), args.knns_);
	fx_timer_stop(fx_root, "dualltree");
	printf("Collecting results....\n");
	fx_timer_start(fx_root, "collecting_results");
  //tree.CollectKNearestNeighborWithMMAP(args.out_file_.c_str());
	tree.CollectKNearestNeighborWithFwrite(args.out_file_.c_str());
	fx_timer_stop(fx_root, "collecting_results");
	unlink(args.out_file_.c_str());	
}

