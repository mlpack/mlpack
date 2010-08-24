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
#include <armadillo>
#include <fastlib/base/arma_compat.h>

void KernelPCA::Init(std::string data_file, index_t knns, 
    index_t leaf_size) {
  arma::mat tmp;
  data::Load(data_file.c_str(), tmp);
//  arma_compat::armaToMatrix(tmp, data_);
  dimension_ = data_.n_rows;
  knns_ = knns;
  allknn_.Init(&data_, &data_, leaf_size, knns);
}

void KernelPCA::Destruct() {
  
}

void KernelPCA::ComputeNeighborhoods() {
  NOTIFY("Building tree...\n");
  fflush(stdout);
  arma::Col<index_t> resulting_neighbors;
  arma::vec  distances;
  NOTIFY("Computing Neighborhoods");
  allknn_.ComputeNeighbors(resulting_neighbors,
                           distances);
  FILE *fp=fopen("allnn.txt", "w");
  if (fp==NULL) {
    FATAL("Unable to open allnn for exporting the results, error %s\n",
           strerror(errno));
  }
  for(index_t i=0; i<resulting_neighbors.n_rows; i++) {
    fprintf(fp, "%lli %lli %lg\n", i / knns_, resulting_neighbors[i],
        distances[i]);
  }
  fclose(fp);
}

template<typename DISTANCEKERNEL>
void KernelPCA::ComputeGeneralKernelPCA(DISTANCEKERNEL kernel,
                                        index_t num_of_eigenvalues,
                                        arma::mat *eigen_vectors,
                                        arma::vec *eigen_values){
  kernel_matrix_.Copy(affinity_matrix_);
  kernel_matrix_.ApplyFunction(kernel);
  arma::vec temp;
  temp.set_size(kernel_matrix_.dimension());
  temp.fill(1.0);
  kernel_matrix_.SetDiagonal(temp);
  kernel_matrix_.EndLoading();
  NOTIFY("Computing eigen values...\n");
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
                               arma::mat &eigen_vectors,
                               arma::vec &eigen_values) {
  std::string vec_file(file);
  vec_file.append(".vectors");
  std::string lam_file(file);
  lam_file.append(".lambdas");
  FILE *fp = fopen(vec_file.c_str(), "w");  
  if (unlikely(fp==NULL)) {
    FATAL("Unable to open file %s, error: %s", vec_file.c_str(), 
           strerror(errno));
  }
  for(index_t i=0; i<eigen_vectors.n_rows; i++) {
    for(index_t j=0; j<eigen_vectors.n_cols; j++) {
       fprintf(fp, "%lg\t", eigen_vectors(i, j));
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  fp = fopen(lam_file.c_str(), "w");  
  if (unlikely(fp==NULL)) {
    FATAL("Unable to open file %s, error: %s", lam_file.c_str(), 
          strerror(errno));
  }
  for(index_t i=0; i<(index_t)eigen_values.n_rows; i++) {
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
// It is not always know if the k nearest neighbors includes the same point with
// 0 distance. It is highly likely in cases where the query tree and the reference tree
// come from different structures that refer to the same dataset. We take care about
// that in this function. 
void KernelPCA::ComputeLLE(index_t num_of_eigenvalues,
                           arma::mat *eigen_vectors,
                           arma::vec *eigen_values) {
  FILE *fp=fopen("allnn.txt", "r");
  if unlikely(fp==NULL) {
    FATAL("Unable to open allnn.txt, error %s\n", strerror(errno));
  }
  uint64 p1, p2;
  double dist;
  uint64 last_point=numeric_limits<uint64>::max();
  arma::vec point;
  point.set_size(dimension_);
  arma::mat neighbor_vals;
  // We initialize everything with knns_ although it is highly likely that
  // if the k nearest neighbors include the same point then we will need
  // a smaller vector knns_-1
  neighbor_vals.set_size(dimension_, knns_-1);
  arma::mat covariance(knns_-1, knns_-1);
  arma::vec ones;
  ones.set_size(knns_-1);
  ones.fill(1);
  arma::vec weights;
  index_t neighbors[knns_];
  index_t i=0;
  kernel_matrix_.Init(data_.n_cols,
                      data_.n_cols,
                      knns_);
  last_point=0;
  point = data_.col(0);
  // Create a unitary matrix
  arma::mat covariance_regularizer;
  covariance_regularizer.set_size(knns_-1, knns_-1);
  covariance_regularizer.zeros();
  for(index_t j=0; j<knns_-1; j++) {
    covariance_regularizer(j, j) = 1;
  }  
    
  while (!feof(fp)) {
    fscanf(fp, "%llu %llu %lg\n", &p1, &p2, &dist);
    if (dist==0) {
      continue;
    }
     if (p1!=last_point) {
      point = data_.col(p1);
      last_point=p1;
      la::MulTransAOverwrite(neighbor_vals, neighbor_vals, &covariance);
      // calculate the covariance matrix trace
      double trace=0;
      for (index_t k=0; k<i; k++) {
        trace+=covariance(k,k);
      }
      la::AddExpert(1e-3*trace, covariance_regularizer, &covariance);
      la::SolveInit(covariance, ones, &weights);
      double sum_weights=0;
      for(index_t k=0; k<weights.n_rows; k++) {
        sum_weights+=weights[k];
      }
      for(index_t k=0; k<weights.n_rows; k++) {
        weights[k]/=sum_weights;
      }
      kernel_matrix_.LoadRow(p1, i, neighbors, weights.memptr());
      i=0;
    }
    neighbor_vals.col(i) = data_.col(p2);
    neighbors[i]=p2;
    la::SubFrom(dimension_, point.memptr(), neighbor_vals.col(i));
    i++;
  }
  kernel_matrix_.Negate();
  kernel_matrix_.SetDiagonal(1.0);
  NONFATAL("Computing eigen values...\n");
  SparseMatrix kernel_matrix1;
  kernel_matrix_.ToFile("i_w.txt");
  kernel_matrix_.EndLoading();
  Sparsem::MultiplyT(kernel_matrix_, &kernel_matrix1);
  kernel_matrix1.ToFile("i_w_i_w.txt");
  kernel_matrix1.EndLoading();
  kernel_matrix1.Eig(num_of_eigenvalues, 
                     "SM", 
                     eigen_vectors,
                     eigen_values, NULL);

}
template<typename DISTANCEKERNEL>
void KernelPCA::ComputeSpectralRegression(DISTANCEKERNEL kernel,
                                 std::map<index_t, index_t> &data_label,
                                 arma::mat *embedded_coordinates, 
                                 arma::vec *eigenvalues) {
  // labels has the label of every point, it is not necessary
  // for every point to have a label, that's why we are using a map and not
  // a vector
  
  // This map has the classes and the points
  std::map<index_t, std::vector<index_t> > classes;
  std::vector<index_t> default_bin;
  // find how many classes we have
  index_t num_of_classes=0;
  std::map<index_t, index_t>::iterator it;
  for(it=data_label.begin(); it!=data_label.end(); it++) {
    if (classes.find(it->second)!=classes.end()) {
      classes[it->second].push_back(it->first);
    } else {
      classes.insert(make_pair(it->second, default_bin));
      classes[it->second].push_back(it->first);
      num_of_classes++;
    }
  }
  
  kernel_matrix_.Copy(affinity_matrix_);
  kernel_matrix_.ApplyFunction(kernel);
  // In the paper it is also called W^{SR}
  SparseMatrix labeled_graph;
  labeled_graph.Init(data_.n_cols, data_.n_cols, 2*knns_);
  // In the paper this is D^{SR}
  SparseMatrix d_sr_mat;
  d_sr_mat.Init(data_.n_cols, data_.n_cols, 1);
  arma::vec d_sr_mat_diag;
  d_sr_mat_diag.set_size(data_.n_cols());
  d_sr_mat_diag.fill(0);
  // Now put the label information
  std::map<index_t, std::vector<index_t> >::iterator it1;
  for(it1=classes.begin(); it1!=classes.end(); it1++) {
    for(index_t i=0; i<(index_t)it1->second.size(); i++) {
      for(index_t j=i+1; j<(index_t)it1->second.size(); j++) {
        d_sr_mat_diag[it1->second[i]]+=1.0/(it1->first+1);
        d_sr_mat_diag[it1->second[j]]+=1.0/(it1->first+1);
        kernel_matrix_.set(it1->second[i], it1->second[j], 1.0);
        labeled_graph.set(it1->second[i], it1->second[j], 1.0/(it1->first+1));
        labeled_graph.set(it1->second[j], it1->second[i], 1.0/(it1->first+1));
      }
    }
  }
  labeled_graph.set_symmetric(true);
  kernel_matrix_.MakeSymmetric();
  kernel_matrix_.EndLoading();
  d_sr_mat.SetDiagonal(d_sr_mat_diag);
  d_sr_mat.set_indices_sorted(true);
  // This is the matrix D that has the sum of the rows or columns
  SparseMatrix d_mat;
  d_mat.Init(data_.n_cols, data_.n_cols, 1);
  d_mat.set_indices_sorted(true);
  arma::vec d_diagonal;
  kernel_matrix_.RowSums(&d_diagonal);
  d_mat.SetDiagonal(d_diagonal);
  d_mat.set_indices_sorted(true);
  SparseMatrix laplacian_mat;
  Sparsem::Subtract(d_mat, kernel_matrix_, &laplacian_mat);
  SparseMatrix d_sr_plus_laplacian_mat;
  Sparsem::Add(d_sr_mat, laplacian_mat, &d_sr_plus_laplacian_mat);
  arma::mat eigenvectors;
  labeled_graph.EndLoading();
  d_sr_plus_laplacian_mat.EndLoading();
  labeled_graph.Eig(d_sr_plus_laplacian_mat, num_of_classes, "LM",
      &eigenvectors, eigenvalues, NULL); 
  // this is the embedding therms alpha as shown in 
  // the paper
  arma::mat alpha_factors;
  la::LeastSquareFitTrans(eigenvectors, data_, &alpha_factors); 
  la::MulTransAInit(alpha_factors, data_,embedded_coordinates);
}

