/**
 * @file orthogonal_init.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the orthogonal matrix initialization method.
 */
#ifndef __MLPACK_METHOS_ANN_INIT_RULES_ORTHOGONAL_INIT_HPP
#define __MLPACK_METHOS_ANN_INIT_RULES_ORTHOGONAL_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize the weight matrix with the orthogonal
 * matrix initialization
 *
 * @tparam MatType Type of matrix (should be arma::mat or arma::spmat).
 * @tparam VecType Type of vector (arma::colvec, arma::mat or arma::sp_mat).
 */
template<typename MatType = arma::mat, typename VecType = arma::colvec>
class OrthogonalInitialization
{
 public:
  /**
   * Initialize the orthogonal matrix initialization rule with the given gain.
   *
   * @param gain The gain value.
   */
  OrthogonalInitialization(const double gain = 1.0) : gain(gain) { }

  /**
   * Initialize the elements of the specified weight matrix with the
   * orthogonal matrix initialization method.
   *
   * @param W Weight matrix to initialize.
   * @param n_rows Number of rows.
   * @return n_cols Number of columns.
   */
  void Initialize(MatType& W, const size_t n_rows, const size_t n_cols)
  {
    MatType V;
    VecType s;

    arma::svd_econ(W, s, V, arma::randu<MatType>(n_rows, n_cols));
    W *= gain;
  }

 private:
  //! The number used as lower bound.
  const double gain;
}; // class OrthogonalInitialization


}; // namespace ann
}; // namespace mlpack

#endif
