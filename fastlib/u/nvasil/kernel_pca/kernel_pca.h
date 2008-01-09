/*
 * =====================================================================================
 * 
 *       Filename:  kernel_pca.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  11/30/2007 08:34:19 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef KERNEL_PCA_H_
#define KERNEL_PCA_H_

#include <string>
#include <strdio.h>
#include <errno.h>
#include <unistd.h>
#include "fastlib/fastlib.h"
#include "fastlib/la/matrix.h"
#include "u/nvasil/binary_dataset.h"
#include "sparse/sparse_matrix.h"

template<typename TREE, bool diagnostic>
class KernelPCA {
	typedef typename TREE::Precision_t   Precision_t;
	typedef TREE                         Tree_t;
  class GaussianKernel {
	 public:
	  void set(double bandwidth) {
		  bandwidth_ = bandwidth;
		}
		double operator()(double distance) {
		  return exp(-distance/bandwidth);
		}
	 private:
		double bandwidth_;
	};	
 public:
	void Init();
  void Destruct();
	void ComputeNeighborhoods(index_t knn);
  void LoadAffinityMatrix();
  template<typename DISTANCEKERNEL>		
	void ComputeGeneralKernelPCA(DISTANCEKERNEL kernel,
			                         index_t num_of_eigenvalues,
			                         Matrix *eigen_vectors,
															 std::vector<double> *eigen_values);
	void ComputeIsomap(index_t num_of_eigenvalues);
	void ComputeLLE(index_t num_of_eigenvalues);
	template<KERNEL>
	void ComputeDiffusionMaps(KERNEL kernel, index_t num_of_eigenvalues);
	void ComputeLaplacialnEigenmaps(index_t);
	void ComputeSpectralRegression(std::string label_file);
	static void SaveToTextFile(std::string file, 
			                       Matrix &eigen_vectors,
		                         std::vector<double> &eigen_values);
	static void SaveToBinaryFile(std::string file, 
			                       Matrix &eigen_vectors,
		                         std::vector<double> &eigen_values);
	
 private:
  Tree_t tree_;
  SparseMatrix kernel_matrix_;	
  SparseMatrix affinity_marix_;
	BinaryDataset<Precision_t> data_; 
};

#include "u/nvasil/kernel_pca/kernel_pca_impl.h"
#endif
