/**  
 * @file kernel_pca.h
 * nvasil@ieee.org
 */
#ifndef KERNEL_PCA_H_
#define KERNEL_PCA_H_

#include <string>
#include <map>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include "fastlib/fastlib.h"
#include "fastlib/la/matrix.h"
#include "fastlib/sparse/sparse_matrix.h"
#include "mlpack/allknn/allknn.h"

class KernelPCATest;
/**
 * KernelPCA class is the main class that implements several spectral methods
 * that are variances of Kernel PCA
 * Most of them share an affinity (proximity) )matrix that is computed
 * with the dual-tree all nearest algorithm. All these methods share
 * this affinity matrix and then define their own kernel matrix based on
 * that. Only distance kernels are supported, kernels that are f(distance) 
 *
 * It computes kernel pca as described by Smola in 
 * the following paper. 
 * @article{scholkopf1999kpc,
 *   title={{Kernel principal component analysis}},
 *   author={Scholkopf, B. and Smola, A. and Muller, K.R.},
 *   journal={Advances in Kernel Methods-Support Vector Learning},
 *   pages={327--352},
 *   year={1999},
 *   publisher={Cambridge MA: MIT Press}
 *   }
 * It also computes Local Linear Embedding as described in the
 * paper
 * @misc{roweis2000ndr,
 *  title={{Nonlinear Dimensionality Reduction by Locally Linear Embedding}},
 *  author={Roweis, S.T. and Saul, L.K.},
 *  journal={Science},
 *  volume={290},
 *  number={5500},
 *  pages={2323--2326},
 *  year={2000}
 *  }
 *
 * Another spectral method implemented here is spectral regression 
 * as described in the paper
 * @article{cai2007sru,
 *  title={{Spectral regression: a unified subspace learning framework for content-based image retrieval}},
 *  author={Cai, D. and He, X. and Han, J.},
 *  journal={Proceedings of the 15th international conference on Multimedia},
 *  pages={403--412},
 *  year={2007},
 *  publisher={ACM Press New York, NY, USA}
 *  }
 * In the future it will also support Laplacian Eigenmaps
 * described here:
 * @misc{belkin2003led,
 *  title={{Laplacian Eigenmaps for Dimensionality Reduction and Data Representation}},
 *  author={Belkin, M. and Niyogi, P.},
 *  journal={Neural Computation},
 *  volume={15},
 *  number={6},
 *  pages={1373--1396},
 *  year={2003},
 *  publisher={MIT Press}
 *  }
 * and Diffusion Maps
 * described here:
 * @phdthesis{lafon:dma,
 *  title={{Diffusion Maps and Geodesic Harmonics}},
 *  author={Lafon, S.},
 *  school={Ph. D. Thesis, Yale University, 2004}
 *  }
 */
class KernelPCA {
 public:
  friend class KernelPCATest;
  /**
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

#include "kernel_pca_impl.h"
#endif
