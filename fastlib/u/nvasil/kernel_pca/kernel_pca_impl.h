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


void KernelPCA::Init(std::string data_file, index_t knns, 
		index_t leaf_size) {
  data::Load(data_file.c_str(), &data_);
	knns_ = knns;
  allknn_.Init(data_, data_, leaf_size, knns);
}

void KernelPCA::Destruct() {
  
}

void KernelPCA::ComputeNeighborhoods() {
  NONFATAL("Building tree...\n");
	fflush(stdout);
	ArrayList<index_t> resulting_neighbors;
	ArrayList<double>  distances;
  allknn_.ComputeNeighbors(&resulting_neighbors,
			                     &distances);
	FILE *fp=fopen("allnn.txt", "w");
	if (fp==NULL) {
	  FATAL("Unable to open allnn for exporting the results, error %s\n",
				   strerror(errno));
	}
  for(index_t i=0; i<resulting_neighbors.size(); i++) {
	  fprintf(fp, "%lli %lli %lg\n", i / knns_, resulting_neighbors[i],
				distances[i]);
	}
  fclose(fp);
}

template<typename DISTANCEKERNEL>
void KernelPCA::ComputeGeneralKernelPCA(DISTANCEKERNEL kernel,
		                                    index_t num_of_eigenvalues,
		                                    Matrix *eigen_vectors,
																				std::vector<double> *eigen_values){
  kernel_matrix_.Copy(affinity_matrix_);
	kernel_matrix_.ApplyFunction(kernel);
	Vector temp;
	temp.Init(kernel_matrix_.dimension());
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

void KernelPCA::ComputeLLE(index_t num_of_eigenvalues,
			                     Matrix *eigen_vectors,
									         std::vector<double> *eigen_values) {
  FILE *fp=fopen("allnn.txt", "r");
  if unlikely(fp==NULL) {
	  FATAL("Unable to open allnn.txt, error %s\n", strerror(errno));
	}
	uint64 p1, p2;
	double dist;
  uint64 last_point=numeric_limits<uint64>::max();
	Vector point;
	point.Init(dimension_);
	Matrix neighbor_vals;
	neighbor_vals.Init(dimension_, knns_);
	Matrix cov(neighbor_vals);
	Vector ones;
	ones.Init(dimension_);
	ones.SetAll(1);
	Vector weights;
	index_t neighbors[knns_];
	index_t i;
	kernel_matrix_.Init(data_.n_rows(),
			                data_.n_rows(),
											knns_);
	while (!feof(fp)) {
		fscanf(fp, "%llu %llu %lg", &p1, &p2, &dist);
		i=0;
    if (p1==last_point) {
		 memcpy(neighbor_vals.GetColumnPtr(i), data_.GetColumnPtr(p2), 
				    sizeof(double)*dimension_);
		 neighbors[i]=p2;
		 la::SubFrom(dimension_, point.ptr(), neighbor_vals.GetColumnPtr(i));
		} else {
		  point.Copy(data_.GetColumnPtr(p1), dimension_);
			last_point=p1;
			i=0;
      la::MulTransBInit(neighbor_vals, neighbor_vals, &cov);
      la::SolveInit(cov, ones, &weights);
      kernel_matrix_.LoadRow(p1, knns_, neighbors, weights.ptr());
			weights.Destruct();
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

