/**
 * @file he_normal.hpp
 * @author kris singh
 *
 * He Normal Intilisation Policy
 * performed by assigning a weights from a N(0, sqrt(2/fan_in)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_HE_NORMAL_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_HE_NORMAL_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

using namespace mlpack::math;
 
namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize weights according to 
 * the formula N(0, sqrt(2/fanin))
 */
class HeNormal
{
 public:
  //Empty Constructor
  HeNormal()
  {}

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
    double var = sqrt(2/ (double)(rows));
    W = arma::zeros(rows, cols);
    W.imbue( [&]() {return RandNormal(0.0, var);});
  }

}; // class HeNormal
} // namespace ann
} // namespace mlpack

#endif
