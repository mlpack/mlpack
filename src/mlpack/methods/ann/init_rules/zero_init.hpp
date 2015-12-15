/**
 * @file zero_init.hpp
 * @author Marcus Edel
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a zero matrix to the weight matrix.
 */
#ifndef __MLPACK_METHODS_ANN_INIT_RULES_ZERO_INIT_HPP
#define __MLPACK_METHODS_ANN_INIT_RULES_ZERO_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize randomly the weight matrix.
 */
class ZeroInitialization
{
 public:
  /**
   *  Create the ZeroInitialization object.
   */
  ZeroInitialization() { /* Nothing to do here */ }

  /**
   * Initialize the elements of the specified weight matrix.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename eT>
  void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
  {
    W = arma::zeros<arma::Mat<eT> >(rows, cols);
  }

  /**
   * Initialize the elements of the specified weight (3rd order tensor).
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename eT>
  void Initialize(arma::Cube<eT>& W,
                  const size_t rows,
                  const size_t cols,
                  const size_t slices)
  {
    W = arma::zeros<arma::Cube<eT> >(rows, cols, slices);
  }
}; // class ZeroInitialization

} // namespace ann
} // namespace mlpack

#endif
