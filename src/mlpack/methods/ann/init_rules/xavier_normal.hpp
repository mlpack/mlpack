/**
 * @file xavier_normal.hpp
 * @author kris singh
 *
 * Xavier Normal Intilisation Policy
 * performed by assigning a weights from a N(0, sqrt(2/fan_in+fan_out)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_XAVIER_NORMAL_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_XAVIER_NORMAL_HPP

#include <mlpack/prereqs.hpp>
 
namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize weights according to 
 * the formula N(0, sqrt(2/fanin+fanout))
 */
class XavierNormal
{
 public:
  //Empty Constructor
  XavierNormal()
  {}

  /**
   * Initialize the elements of the specified weight matrix.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */ 
  template<typename eT>
  void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols, const size_t seed=21)
  {
    W = arma::zeros(rows, cols);
    std::mt19937 engine;  // Mersenne twister random number engine
    std::srand(seed);
    double var = sqrt(2/ (static_cast<double>(rows)+static_cast<double>(cols)));
    std::normal_distribution<double> distr(0, var);
    W.imbue( [&]() {return distr(engine); });
  }

}; // class XavierNormal
} // namespace ann
} // namespace mlpack

#endif
