/**
 * @file methods/ann/init_rules/const_init.hpp
 * @author Sumedh Ghaisas
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a matrix of all constant values to the weight
 * matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_CONST_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_CONST_INIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize weight matrix with constant values.
 */
class ConstInitialization
{
 public:
  /**
   *  Create the ConstantInitialization object.
   */
  ConstInitialization(const double initVal = 0) : initVal(initVal)
  { /* Nothing to do here */ }

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
    if (W.is_empty())
      W.set_size(rows, cols);

    W.fill(initVal);
  }

  /**
   * Initialize the elements of the specified weight matrix.
   *
   * @param W Weight matrix to initialize.
   */
  template<typename eT>
  void Initialize(arma::Mat<eT>& W)
  {
    if (W.is_empty())
      Log::Fatal << "Cannot initialize an empty matrix." << std::endl;

    W.fill(initVal);
  }

  /**
   * Initialize the elements of the specified weight (3rd order tensor).
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param slices Number of slices.
   */
  template<typename eT>
  void Initialize(arma::Cube<eT>& W,
                  const size_t rows,
                  const size_t cols,
                  const size_t slices)
  {
    if (W.is_empty())
      W.set_size(rows, cols, slices);

    W.fill(initVal);
  }

  /**
   * Initialize the elements of the specified weight (3rd order tensor).
   *
   * @param W Weight matrix to initialize.
   */
  template<typename eT>
  void Initialize(arma::Cube<eT>& W)
  {
    if (W.is_empty())
      Log::Fatal << "Cannot initialize an empty cube." << std::endl;

    W.fill(initVal);
  }

  //! Get the initialization value.
  double const& InitValue() const { return initVal; }
  //! Modify the initialization value.
  double& initValue() { return initVal; }

 private:
  //! Value to be initialized with
  double initVal;
}; // class ConstInitialization

} // namespace ann
} // namespace mlpack

#endif
