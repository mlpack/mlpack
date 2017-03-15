/**
 * @file xavier_uniform.hpp
 * @author kris singh
 *
 * Xavier Uniform Intilisation Policy
 * performed by assigning a weights from a U(0, 2/fan_in + fan_out)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_XAVIER_UNIFORM_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_XAVIER_UNIFORM_HPP

#include <mlpack/prereqs.hpp>
 
namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize weights according to 
 * the formula U[-12/(fan_in+fan_out), 12/(fan_in+fan_out)]
 * Refrence http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
 */
class XavierUniform
{
 public:
  //Empty Constructor
  XavierUniform()
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
    std::mt19937 engine;  // Mersenne twister random number engine
    double a = sqrt(12/ rows+cols);
    std::uniform_real_distribution<double> distr(-a, +a);
    W = arma::zeros(rows, cols);
    W.imbue( [&]() {return distr(engine); });
  }

}; // class XavierUniform
} // namespace ann
} // namespace mlpack

#endif
