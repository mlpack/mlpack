/**
 * @file nca.hpp
 * @author Ryan Curtin
 *
 * Declaration of NCA class (Neighborhood Components Analysis).
 */
#ifndef __MLPACK_METHODS_NCA_NCA_HPP
#define __MLPACK_METHODS_NCA_NCA_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace nca /** Neighborhood Components Analysis. */ {

/**
 * An implementation of Neighborhood Components Analysis, both a linear
 * dimensionality reduction technique and a distance learning technique.  The
 * method seeks to improve k-nearest-neighbor classification on a dataset by
 * scaling the dimensions.  The method is nonparametric, and does not require a
 * value of k.  It works by using stochastic ("soft") neighbor assignments and
 * using optimization techniques over the gradient of the accuracy of the
 * neighbor assignments.
 *
 * For more details, see the following published paper:
 *
 * @code
 * @inproceedings{Goldberger2004,
 *   author = {Goldberger, Jacob and Roweis, Sam and Hinton, Geoff and
 *       Salakhutdinov, Ruslan},
 *   booktitle = {Advances in Neural Information Processing Systems 17},
 *   pages = {513--520},
 *   publisher = {MIT Press},
 *   title = {{Neighbourhood Components Analysis}},
 *   year = {2004}
 * }
 * @endcode
 */
template<typename Kernel>
class NCA
{
 public:
  /**
   * Construct the Neighborhood Components Analysis object.  This simply stores
   * the reference to the dataset, before the actual optimization is performed.
   *
   * @param dataset Input dataset.
   */
  NCA(const arma::mat& dataset, const arma::uvec& labels);

  /**
   * Perform Neighborhood Components Analysis.  The output distance learning
   * matrix is written into the passed reference.
   *
   * @param output_matrix Covariance matrix of Mahalanobis distance.
   */
  void LearnDistance(arma::mat& output_matrix);

 private:
  const arma::mat& dataset_;
  const arma::uvec& labels_;
};

}; // namespace nca
}; // namespace mlpack

// Include the implementation.
#include "nca_impl.hpp"

#endif
