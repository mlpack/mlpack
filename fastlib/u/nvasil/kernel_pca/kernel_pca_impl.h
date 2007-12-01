/*
 * =====================================================================================
 * 
 *       Filename:  kernel_pca_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  11/30/2007 09:03:12 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#define TEMPLATE__           \
template<typename TREE, bool diagnostic>
#define KERNELPCA__ KernelPCA<TREE, diagnostic>

TEMPLATE__
void KERNELPCA::Init(std::string data_file, 
		                 std::string index_file) {
	if (index_file.empty()) {
    data_.Init(data_file);
	} else {
	  data_.Init(data_file, index_file);
	}
	tree_.Init(&data_);
}

TEMPLATE__
void KERNELPCA::Destruct() {
  tree_.Destruct();
  data_.Destruct();	
}

TEMPLATE__
void KERNELPCA::ComputeAffinity(index_t knns) {
    fx_timer_start(fx_root, "build_tree")
	  test_tree.BuildDepthFirst();
		fx_timer_start(fx_root, "duall_tree");	
	  train_tree.AllNearestNeighbors(tree_.get_parent(), knns);
	  fx_timer_stop(fx_root, "duall_tree");
}

TEMPLATE__
template<typename DISTANCEKERNEL>
void KERNELPCA::ComputeGeneralKernelPCA(DISTANCEKERNEL kernel, 
		                                    index_t num_of_eigenvalues){


}



