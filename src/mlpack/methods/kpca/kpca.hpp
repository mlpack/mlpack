/**
 * @file kpca.hpp
 * @author Ajinkya Kale
 *
 * Defines the KPCA class to perform Kernel Principal Components Analysis on the
 * specified data set.
 */
#ifndef __MLPACK_METHODS_KPCA_KPCA_HPP
#define __MLPACK_METHODS_KPCA_KPCA_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>

namespace mlpack {
namespace kpca {

template <typename KernelType>
class KPCA
{
 public:
  KPCA(const KernelType kernel = KernelType(),
       const bool centerData = true,
       const bool scaleData = false);

  /* Return whether or not this KPCA object will center the data when KPCA
   *  is performed.
   */
  bool CenterData() const
  {
    return centerData_;
  }

  /* Modify whether or not this KPCA object will center the data when KPCA
   * is performed.
   */
  bool& CenterData()
  {
    return centerData_;
  }

  /* Return whether or not this KPCA object will scale(by standard deviation) the data when KPCA
   *  is performed.
   */
  bool ScaleData() const
  {
    return scaleData_;
  }

  /* Modify whether or not this KPCA object will scale(by standard deviation) the data when KPCA
   * is performed.
   */
  bool& ScaleData()
  {
    return scaleData_;
  }

  /**
   * Apply Kernel Principal Component Analysis to the provided data set.
   *
   * @param data - Data matrix
   * @param transformedData - Data with PCA applied
   * @param eigVal - contains eigen values in a column vector
   * @param coeff - PCA Loadings/Coeffs/EigenVectors
   */
  void Apply(const arma::mat& data, arma::mat& transformedData, arma::vec&
             eigVal, arma::mat& coeff);

  /**
   * Apply Kernel Principal Component Analysis to the provided data set.
   *
   * @param data - Data matrix
   * @param transformedData - Data with PCA applied
   * @param eigVal - contains eigen values in a column vector
   */
  void Apply(const arma::mat& data, arma::mat& transformedData,
             arma::vec& eigVal);

  /**
   * Apply Dimensionality Reduction using Kernel Principal Component Analysis
   * to the provided data set.
   *
   * @param data - M x N Data matrix
   * @param newDimension - matrix consisting of N column vectors,
   * where each vector is the projection of the corresponding data vector
   * from data matrix onto the basis vectors contained in the columns of
   * coeff/eigen vector matrix with only newDimension number of columns chosen.
   */
  void Apply(arma::mat& data, const int newDimension);

  /*
   * Delete KPCA object
   */
  //~KPCA();

 private:
   bool centerData_;
   bool scaleData_;
   KernelType kernel_;

}; // class KPCA

}; // namespace kpca
}; // namespace mlpack

// Include implementation.
#include "kpca_impl.hpp"

#endif // __MLPACK_METHODS_KPCA_HPP
