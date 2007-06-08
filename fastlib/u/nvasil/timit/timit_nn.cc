/*
 * =====================================================================================
 *
 *       Filename:  timit_nn.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  06/07/2007 02:05:14 PM EDT
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
#include "u/nvasil/dataset/binary_dataset.h"
#include "u/nvasil/tree/binary_kd_tree_mmapmm.h"

struct Parameters {
	std::string train_file_;
	std::string test_file_;
	std::string out_file_;	
	index_t knns_;
	std::string memory_file_;
  BinaryDataset<float32> train_data_;	
  BinaryDataset<float32> test_data_;
	uint64 capacity_;
};
std::string Usage();
template<typename TREE>
void DuallTreeAllNearestNeighborsSpecializedForKnn(Parameters &args);

int main(int argc, char *argv[]) {
	Parameters args;
	// initialize command line parameter
	fx_init(argc, argv);
	if (fx_param_exists(NULL, "help")) {
	  printf("%s\n", Usage().c_str());
		return -1;
	}
  args.train_file_=fx_param_str(NULL, "train_file", "");
  args.test_file_=fx_param_str(NULL, "test_file", "");
  args.knns_=fx_param_int(NULL, "knns",5);
	args.out_file_=fx_param_str(NULL, "out_file", "allnn");
  args.memory_file_=fx_param_str(NULL, "memory_file", "temp_mem");
	args.capacity_=fx_param_int(NULL, "capacity", 16777216);
 	if (sizeof(index_t)==sizeof(int32)) {
	  NONFATAL("index_t is int32, good for small scale problems");
	} else {
		if (sizeof(index_t)==sizeof(int64)) {
	    NONFATAL("index_t is int64, good for large scale problems");
		}	 
	}
   
	NONFATAL("Creating swap file...\n");
	mmapmm::MemoryManager<false>::allocator_ = 
		    new mmapmm::MemoryManager<false>();
    mmapmm::MemoryManager<false>::
			  allocator_->set_capacity(args.capacity_);
		mmapmm::MemoryManager<false>::
			  allocator_->set_pool_name(args.memory_file_);
		mmapmm::MemoryManager<false>::allocator_->Init();
    DuallTreeAllNearestNeighborsSpecializedForKnn<
			    BinaryKdTreeMMAPMMKnnNode_t>(args);  
	fx_format_result(fx_root, "success", "%d", 1);
	fx_done();
}


template<typename TREE>
void DuallTreeAllNearestNeighborsSpecializedForKnn(Parameters &args) {
  TREE train_tree;
	TREE test_tree;
	NONFATAL("Procceding with the specialized method for knn node..\n");
  if (args.train_file_!=args.test_file_) {
	  args.train_data_.Init(args.train_file_);
	  train_tree.Init(&args.train_data_);
	  fx_timer_start(fx_root, "train_tree_build");	
    NONFATAL("Building the training (reference) tree...\n");
	  fflush(stdout);
	  train_tree.BuildDepthFirst();
	  fx_timer_stop(fx_root, "train_tree_build");
    NONFATAL("Memory usage: %llu\n",
	          (unsigned long long)TREE::Allocator_t::allocator_->get_usage());
	  NONFATAL("Training (Reference) tree \n %s\n", train_tree.Statistics().c_str());
	  args.train_data_.Destruct();
	  args.test_data_.Init(args.test_file_);
	  test_tree.Init(&args.test_data_);
	  test_tree.set_knns(args.knns_);
	  NONFATAL("Building the test (query) tree...\n");
	  fflush(stdout);
	  fx_timer_start(fx_root, "test_tree_build");
	  test_tree.BuildDepthFirst();
	  fx_timer_stop(fx_root, "test_tree_build");
    NONFATAL("Memory usage: %llu\n",
	          (unsigned long long)TREE::Allocator_t::allocator_->get_usage());
	  NONFATAL("Test (Query) tree \n %s\n", test_tree.Statistics().c_str());
	  args.test_data_.Destruct();
    if (train_tree.get_dimension()!=test_tree.get_dimension()) {
	    FATAL("Train set has different dimension %i than Test set %i\n",
			  	  args.train_data_.get_dimension(),
			      args.test_data_.get_dimension());
		}
	  NONFATAL("Computing all nearest neighbors...\n");
	  fflush(stdout);  
	  fx_timer_start(fx_root, "dualltree");	
	  train_tree.AllNearestNeighbors(test_tree.get_parent(), args.knns_);
	  fx_timer_stop(fx_root, "dualltree");
	  NONFATAL("Collecting results....\n");
	  fx_timer_start(fx_root, "collecting_results");
	  test_tree.CollectKNearestNeighborWithFwriteText(args.out_file_.c_str());
	  fx_timer_stop(fx_root, "collecting_results");
	} else {
	  NONFATAL("Training and test tree are the same\n" );
		args.train_data_.Init(args.test_file_);
	  train_tree.Init(&args.train_data_);
    train_tree.set_knns(args.knns_);
	  fx_timer_start(fx_root, "train_tree_build");	
    NONFATAL("Building the training (reference) tree...\n");
	  fflush(stdout);
	  train_tree.BuildDepthFirst();
	  fx_timer_stop(fx_root, "train_tree_build");
    NONFATAL("Memory usage: %llu\n",
	          (unsigned long long)TREE::Allocator_t::allocator_->get_usage());
	  NONFATAL("Training (Reference) tree \n %s\n", train_tree.Statistics().c_str());
	  args.train_data_.Destruct();
    NONFATAL("Computing all nearest neighbors...\n");
	  fflush(stdout);  
	  fx_timer_start(fx_root, "dualltree");	
	  train_tree.AllNearestNeighbors(train_tree.get_parent(), args.knns_);
	  fx_timer_stop(fx_root, "dualltree");
	  NONFATAL("Collecting results....\n");
	  fx_timer_start(fx_root, "collecting_results");
	  train_tree.CollectKNearestNeighborWithFwriteText(args.out_file_.c_str());
	  fx_timer_stop(fx_root, "collecting_results");
	}
}
std::string Usage() {
  std::string ret =
		string("Computing all k-nearest neighbors with dual tree method...\n") +
		string("timit_nn --option=value\n")+
	  string("--train_file :  the dataset that contains the training data (reference)\n")+
	  string("--test_file  :  the dataset that contains the test data (query)\n")+
		string("--out_file   :  stores the results in a text file, usually big\n")+
		string("--memory_file:  the file wher the tree will be stored\n")+
		string("--knns       :  number of neighbors default is 5\n")+
		string("--capacity   :  the capacity of the memory file, keep it big enough\n");
	 return ret;
}

