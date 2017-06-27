/**
 * @file func_sq.hpp
 * @author Chenzhe Diao
 *
 * Square loss function: \f$ x-> 0.5 * || Ax - b ||_2^2 \f$.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_FUNC_SQ_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_FUNC_SQ_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

class FuncSq
{
 public:
  FuncSq(const arma::mat A, const arma::mat b ) : A(A), b(b)
  {/* Nothing to do. */}

  double Evaluate(const arma::mat& coords)
  {
      arma::vec r = A*coords - b;
      arma::mat y = (r.t() * r) * 0.5;
      return y(0,0);
  }

  void Gradient(const arma::mat& coords, arma::mat& gradient)
  {
      arma::vec r = A*coords - b;
      gradient = A.t() * r;
  }


  arma::mat MatrixA() const {return A;}
  arma::mat& MatrixA() {return A;}

  arma::vec Vectorb() const { return b; }
  arma::vec& Vectorb() { return b; }

 private:
  arma::mat A;  // matrix
  arma::vec b;  // vector

};

} // namespace optimization
} // namespace mlpack

#endif
