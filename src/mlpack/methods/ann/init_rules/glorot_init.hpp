/**
 * @file glorot_init.hpp
 * @author Prabhat Sharma
 *
 * Definition and implementation of the Glorot initialization method. This
 * initialization rule initialize the weights to maintain activation variances
 * and back-propagated gradients variance as one moves up or down the network.
 *
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings{pmlr-v9-glorot10a,
 * title={Understanding the difficulty of training deep feedforward neural networks},
 * author={Xavier Glorot and Yoshua Bengio},
 * booktitle={Proceedings of the Thirteenth International Conference on Artificial
 *              Intelligence and Statistics},
 * year={2010}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_GLOROT_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_GLOROT_INIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

using namespace mlpack::math;

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize the weight matrix with the Glorot Initialization
 * method. The method is defined by
 *
 * @f{eqnarray*}{
 * \Var[w_i] &=& \frac{2}{n_i + n_{i+1}} \\
 * w_i ~ \U{-\frac{\sqrt(6)}{\sqrt(n_i + n_{i+1})}, \frac{\sqrt(6)}{\sqrt(n_i + n_{i+1})}}
 * @f}
 * Where n_{i+1} is the number of neurons in the outgoing layer, n_i represents the
 * number of neurons in the ingoing layer
 * Here Normal Distribution may also be used if needed
 */
template<typename Distribution = bool>
class GlorotInitialization
{
 public:
  /**
   * Initialize
   */
  GlorotInitialization(const Distribution& uniform = Distribution()) :
    uniform(uniform)
  {
    // Nothing to do here.
  }

  /**
   * Initialize the elements weight matrix.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename eT>
  void Initialize(arma::Mat<eT>& W,
                  const size_t rows,
                  const size_t cols)
  {
     double_t a = sqrt(6)/sqrt(rows + cols); // limit of  distribution

    if (W.is_empty())
    {
      W = arma::mat(rows, cols);
    }

    if (!uniform)
      W.imbue( [&]() { return arma::as_scalar(Random(-a, a)); } );
    else
    {
      double_t var = (double)(2)/(rows + cols);
      W.imbue([&]() { return arma::as_scalar(RandNormal(0.0, var)); });
    }
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor with glorot initialization method
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param slice Numbers of slices.
   */
  template<typename eT>
  void Initialize(arma::Cube<eT>& W,
                  const size_t rows,
                  const size_t cols,
                  const size_t slices)
  {
    if (W.is_empty())
    {
      W = arma::cube(rows, cols, slices);
    }
    for (size_t i = 0; i < slices; i++)
      Initialize(W.slice(i), rows, cols);
  }

 private:
  //! Mode used i.e. Uniform or Normal
  bool uniform;
}; // class GlorotInitialization

} // namespace ann
} // namespace mlpack

#endif
