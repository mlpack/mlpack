/**
 * @file he_init.hpp
 * @author kris singh
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a weights from a uniform distribution between 2/fan_in + fan_out.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_HE_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_HE_INIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

#include "gaussian_init.hpp"

 
namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

class HeNormal
{};
class HeUniform
{};

/**
 * Base class for He Intilisation Policy
 * Refrence 
 * Kaiming He et al. (2015):
 * Delving deep into rectifiers: Surpassing human-level performance on
 * imagenet classification. arXiv preprint arXiv:1502.01852.
 */
class HeInit
{
 public:
  //Empty Constructor
  //Scaling factor should be set according to the activation function
  HeInit(const size_t scalingFactor = 1):
  scalingFactor(scalingFactor)
  {}

  template<typename InitializerType, typename eT>
  typename std::enable_if<std::is_same<InitializerType, HeNormal>::value, void>::type
  Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
  {
    double var = sqrt(2 / (double)(rows));
    GaussianInitialization init(0, var);
    init.Initialize(W, rows, cols);
    W = scalingFactor * W;
  }


  template<typename InitializerType, typename eT>
  typename std::enable_if<std::is_same<InitializerType, HeUniform>::value, void>::type
  Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
  {
    double var = sqrt(2 / static_cast<double>(rows));
    W = arma::zeros(rows, cols);
    W.imbue( [&]() {return Random(-var, var);});
    W = scalingFactor * W;
  }
  /**
   * Initialize randomly the elements of the specified weight 3rd order tensor.
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
    W = arma::Cube<eT>(rows, cols, slices);

    for (size_t i = 0; i < slices; i++)
      Initialize(W.slice(i), rows, cols);
  }

private:
  const size_t scalingFactor;

}; // class RandomInitialization
} // namespace ann
} // namespace mlpack

#endif
