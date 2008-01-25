/**  
 * @file kernel_pca.h
 * It computes kernel pca as described by Smola in 
 * the following paper. 
 *
 * It also computes Local Linear Embedding as described in the
 * paper
 *
 *
 * Another spectral method implemented here is spectral regression 
 * as described in the paper
 *
 * In the future it will also support Laplacian Eigenmaps
 * described here:
 * 
 * and Diffusion Maps
 * described here
 */
#ifndef KERNEL_PCA_H_
#define KERNEL_PCA_H_

#include <string>
#include <map>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include "fastlib/fastlib.h"
#include "la/matrix.h"
#include "sparse/sparse_matrix.h"
#include "u/nvasil/allknn/allknn.h"

class KernelPCATest;
/*
 * KernelPCA class is the main class that implements several spectral methods
 * that are variances of Kernel PCA
 * Most of them share an affinity (proximity) )matrix that is computed
 * with the dual-tree all nearest algorithm. All these methods share
 * this affinity matrix and then define their own kernel matrix based on
 * that. Only distance kernels are supported, kernels that are f(distance) 
 */
class KernelPCA {
 public:
  friend class KernelPCATest;
  /*
   * Example of a kernel. It should be a class overloading the 
   * operator() 
   * Here we have the gaussian kernel
   */
  class GaussianKernel {
   public:
    void set(double bandwidth) {
      bandwidth_ = bandwidth;
    }
    double operator()(double distance) {
      return exp(-distance/bandwidth_);
    }
   private:
    double bandwidth_;
  };  
  
  ~KernelPCA() {
    Destruct();
  }
  /**
   * Initializer
   * data_file: contains the data in a csv file
   * knns:      the number of the k-neighborhood for the affinity(proximity)
   *            matrix
   * leaf_size: maximun number of points on a leaf
   *            
   */
  void Init(std::string data_file, index_t knns, 
      index_t leaf_size);
  void Destruct();
  /**
   * Generates the neighborhoods with the dual tree all nearest 
   * neighbors algorithm and stores them to a file allnn.txt
   */
  void ComputeNeighborhoods();
  /**
   * Loads the results to the sparse affinity matrix
   */
  void LoadAffinityMatrix();
  /**
   * Estimates the local bandwidth by taking tha average k-nearest
   * neighbor distance
   */
  void EstimateBandwidth(double *bandwidth);
  /**
   * A simple way to save the results to a file
   */
  static void SaveToTextFile(std::string file, 
                             Matrix &eigen_vectors,
                             Vector &eigen_values);
  static void SaveToBinaryFile(std::string file, 
                             Matrix &eigen_vectors,
                             Vector &eigen_values);
  /**
   * After computing the neighboroods and loading
   * the affinity matrix call this function
   * to compute the num_of_eigenvalues first components
   * of kernel pca
   */
  template<typename DISTANCEKERNEL>    
  void ComputeGeneralKernelPCA(DISTANCEKERNEL kernel,
                               index_t num_of_eigenvalues,
                               Matrix *eigen_vectors,
                               Vector *eigen_values);
  /**
   * Not implemented yet
   */
  void ComputeIsomap(index_t num_of_eigenvalues);
  /**
   * Local Linear Embedding. Note that you have to call first
   * ComputeNeighborhoods and then Load Affinity Matrix
   */
  void ComputeLLE(index_t num_of_eigenvalues,
                  Matrix *eigen_vectors,
                  Vector *eigen_values);
  /**
   * Not implemented yet
   */
  template<typename DISTANCEKERNEL>
  void ComputeDiffusionMaps(DISTANCEKERNEL kernel, index_t num_of_eigenvalues);
  /**
   * Not implemented yet
   */
  void ComputeLaplacialnEigenmaps(index_t);
  /**
   * Spectral Regression 
   * std::map<index_t, index_t> &data_label: For some data points
   * it assign numerical labels
   */
  template<typename DISTANCEKERNEL>
  void ComputeSpectralRegression(DISTANCEKERNEL kernel,
                                 std::map<index_t, index_t> &data_label,
                                 Matrix *embedded_coordinates, 
                                 Vector *eigenvalues);
    
 private:
  AllkNN allknn_;
  index_t knns_;
  Matrix data_;
  SparseMatrix kernel_matrix_;  
  SparseMatrix affinity_matrix_;
  index_t dimension_;
};

#include "u/nvasil/kernel_pca/kernel_pca_impl.h"
#endif
