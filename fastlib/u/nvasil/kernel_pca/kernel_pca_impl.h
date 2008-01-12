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


void KernelPCA::Init(std::string data_file, 
		                 std::string index_file) {
	if (index_file.empty()) {
    data_.Init(data_file);
	} else {
	  data_.Init(data_file, index_file);
	}
	dimension_=data_.get_dimension();
	tree_.Init(&data_);
  mmapmm::MemoryManager<false>::allocator_ = 
		    new mmapmm::MemoryManager<false>();
  mmapmm::MemoryManager<false>::allocator_->Init();
}

void KernelPCA::Destruct() {
  tree_.Destruct();
  data_.Destruct();	
	unlink("allnn.txt");
	if (mmapmm::MemoryManager<false>::allocator_ != NULL) {
	  delete mmapmm::MemoryManager<false>::allocator_;
    mmapmm::MemoryManager<false>::allocator_=NULL;
	}
}

void KernelPCA::ComputeNeighborhoods(index_t knns) {
  NONFATAL("Building tree...\n");
	fflush(stdout);
	tree_.set_knns(knns);
	tree_.BuildDepthFirst();
	NONFATAL("Memory usage: %llu\n",
	          (unsigned long long)Tree_t::Allocator_t::allocator_->get_usage());
  NONFATAL("Tree Statistics\n %s\n", tree_.Statistics().c_str());
  NONFATAL("Computing all nearest neighbors...\n");
  fflush(stdout);  
	tree_.AllNearestNeighbors(tree_.get_parent(), knns);
  NONFATAL("Collecting results....\n");
	tree_.CollectKNearestNeighborWithFwriteText("allnn.txt");
}

template<typename DISTANCEKERNEL>
void KernelPCA::ComputeGeneralKernelPCA(DISTANCEKERNEL kernel,
		                                    index_t num_of_eigenvalues,
		                                    Matrix *eigen_vectors,
																				std::vector<double> *eigen_values){
  kernel_matrix_.Copy(affinity_matrix_);
	kernel_matrix_.ApplyFunction(kernel);
	Vector temp;
	temp.Init(kernel_matrix_.get_dimension());
	temp.SetAll(1.0);
	kernel_matrix_.SetDiagonal(temp);
	kernel_matrix_.EndLoading();
	NONFATAL("Computing eigen values...\n");
  kernel_matrix_.Eig(num_of_eigenvalues, 
			               "LM", 
										 eigen_vectors,
										 eigen_values, NULL);
}

void KernelPCA::LoadAffinityMatrix() {
  affinity_matrix_.Init("allnn.txt");
  affinity_matrix_.MakeSymmetric();
}

void KernelPCA::SaveToTextFile(std::string file, 
                               Matrix &eigen_vectors,
		                           std::vector<double> &eigen_values) {
	std::string vec_file(file);
  vec_file.append(".vectors");
	std::string lam_file(file);
  lam_file.append(".lambdas");
  FILE *fp = fopen(vec_file.c_str(), "w");	
	if (unlikely(fp==NULL)) {
	  FATAL("Unable to open file %s, error: %s", vec_file.c_str(), 
				   strerror(errno));
	}
	for(index_t i=0; i<eigen_vectors.n_rows(); i++) {
	  for(index_t j=0; j<eigen_vectors.n_cols(); j++) {
		   fprintf(fp, "%lg\t", eigen_vectors.get(i, j));
		}
		fprintf(fp, "\n");
	}
  fclose(fp);
  fp = fopen(lam_file.c_str(), "w");	
	if (unlikely(fp==NULL)) {
	  FATAL("Unable to open file %s, error: %s", lam_file.c_str(), 
				  strerror(errno));
	}
	for(index_t i=0; i<(index_t)eigen_values.size(); i++) {
	  fprintf(fp, "%lg\n", eigen_values[i]);
	}
	fclose(fp);
}

void KernelPCA::EstimateBandwidth(double *bandwidth) {
  FILE *fp=fopen("allnn.txt", "r");
	if unlikely(fp==NULL) {
	  FATAL("Unable to open allnn.txt, error %s\n", strerror(errno));
	}
	uint64 p1, p2;
	double dist;
	double mean=0;
	uint64 count=0;
	while (!feof(fp)) {
		fscanf(fp, "%llu %llu %lg", &p1, &p2, &dist);
		mean+=dist;
		count++;
	}
	*bandwidth=mean/count;
}

void KernelPCA::ComputeLLE(index_t knns,
		                       index_t num_of_eigenvalues,
			                     Matrix *eigen_vectors,
									         std::vector<double> *eigen_values);
{
  FILE *fp=fopen("allnn.txt");
  if unlikely(fp==NULL) {
	  FATAL("Unable to open allnn.txt, error %s\n", strerror(errno));
	}
	uint64 p1, p2;
	double dist;
	double mean=0;
  uint64 last_point=numeric_limits<uint64>::max();
	Vector point;
	point.Init(dimension_);
	Matriix neighbors;
	neighbor_vals.Init(dimension_, knns);
	Matrix cov(neighbors);
	Vector ones;
	ones.Init(dimension_);
	ones.SetAll(1);
	Vector weights;
	index_t neighbors[knns];
	index_t i;
	kernel_matrix_.Init(data_.get_num_of_points(),
			                data_.get_num_of_point());
	while (!feof(fp)) {
		fscanf(fp, "%llu %llu %lg", &p1, &p2, &dist);
		i=0;
    if (p1==last_point) {
		 memcpy(neighbor_vals.GetColumnPtr(i), data_.At(p2), 
				    sizeof(double)*dimension_);
		 neighbors[i]=p2;
		 la::SubFrom(dimension_, point.ptr(), neighbor_vals.GetColumnPtr(i));
		} else {
		  point.Copy(data_.At());
			last_point=p1;
			i=0;
      la::MulTransBInit(neighbor_vals, neighbor_vals, &cov);
      la::SolveInit(cov, ones, &weights);
      kernel_matrix_.LoadRow(p1, neighbors,weights.ptr());
			weights.Destruct()
		}
	}
  kernel_matrix_.Negate();
	kernel_matrix_.SetDiagonal(1.0);
  NONFATAL("Computing eigen values...\n");
  kernel_matrix_.Eig(num_of_eigenvalues, 
			               "SM", 
										 eigen_vectors,
										 eigen_values, NULL);

}

