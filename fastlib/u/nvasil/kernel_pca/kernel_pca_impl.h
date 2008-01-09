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
	unlink("temp.txt");
}

TEMPLATE__
void KERNELPCA::ComputeNeighborhoods(index_t knns) {
  NONFATAL("Building tree...\n");
	fflush(stdout);
	fx_timer_start(fx_root, "build_tree")
	tree_.BuildDepthFirst();
	NONFATAL("Memory usage: %llu\n",
	          (unsigned long long)Tree_t::Allocator_t::allocator_->get_usage());
  NONFATAL("Computing all nearest neighbors...\n");
  fflush(stdout);  
	fx_timer_start(fx_root, "duall_tree");	
	tree_.AllNearestNeighbors(tree_.get_parent(), knns);
	fx_timer_stop(fx_root, "duall_tree");
  NONFATAL("Collecting results....\n");
	fx_timer_start(fx_root, "collecting_results");
	train_tree.CollectKNearestNeighborWithFwriteText("temp.txt");
	fx_timer_stop(fx_root, "Collecting results, saving to file");
}

TEMPLATE__
template<typename DISTANCEKERNEL>
void KERNELPCA::ComputeGeneralKernelPCA(DISTANCEKERNEL kernel,
		                                    index_t num_of_eigenvalues,
		                                    Matrix *eigen_vectors,
																				std::vector<double> eigen_values){
  kernel_matrix_.Copy(affinity_matrix_);
	kernel_matrix_.ApplyFunction(kernel);
  kernel_matrix_.Eig(num_of_eigenvalues, 
			               "LM", 
										 eigen_vectors,
										 eigvalues, NULL);
}

TEMPLATE__
void KERNELPCA::LoadAffinityMatrix() {
  affinity_matrix_.Init("temp.txt");
	affinity_matrix_.MakeSymmetric();
}

TEMPLATE__
static void KERNELPCA::SaveToTextFile(std::string file, 
			                                Matrix &eigen_vectors,
		                                  std::vector<double> &eigen_values) {
  FILE *fp = fopen(file.append("vectors").c_str());	
	if (unlikely(fp==NULL)) {
	  FATAL("Unable to open file %s, error: %s", file.append("vectors").c_str(), 
				   strerr(errno));
	}
	for(index_t i=0; i<eigen_vectors.n_rows()) {
	  for(index_t j=0; j<eigen_vectors.n_cols()) {
		   fprintf(fp, "%lg\t", eigen_values[i][j]);
		}
		fprintf(fp, "\n");
	}
  fclose(fp);
  fp = fopen(file.append("lambdas").c_str());	
	if (unlikely(fp==NULL)) {
	  FATAL("Unable to open file %s, error: %s", file.append(lambdas).c_str(), 
				  strerr(errno));
	}
	for(index_t i=0; i<eigen_values.size(); i++) {
	  fprintf(fp, "%lg", eigen_values[i]);
	}
	fclose(fp);
}

