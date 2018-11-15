/**
 * @file func_sq.hpp
 * @author Chenzhe Diao
 *
 * Square loss function: \f$ x-> 0.5 * || Ax - b ||_2^2 \f$.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FW_FUNC_SQ_HPP
#define ENSMALLEN_FW_FUNC_SQ_HPP

namespace ens {

/**
 * Square loss function \f$ f(x) = 0.5 * ||Ax - b||_2^2 \f$.
 *
 * Contains matrix \f$ A \f$ and vector \f$ b \f$.
 */
class FuncSq
{
 public:
  /**
   * Construct the square loss function.
   *
   * @param A matrix A.
   * @param b vector b.
   */
  FuncSq(const arma::mat& A, const arma::vec& b) : A(A), b(b)
  {/* Nothing to do. */}

  /**
   * Evaluation of the function.
   * \f$ f(x) = 0.5 * ||Ax - b||_2^2 \f$
   *
   * @param coords vector x.
   * @return \f$ f(x) \f$.
   */
  double Evaluate(const arma::mat& coords)
  {
    arma::vec r = A * coords - b;
    return arma::dot(r, r) * 0.5;
  }

  /**
   * Gradient of square loss function.
   * \f$ \nabla f(x) = A^T(Ax - b) \f$
   *
   * @param coords input vector x.
   * @param gradient output gradient vector.
   */
  void Gradient(const arma::mat& coords, arma::mat& gradient)
  {
    arma::vec r = A * coords - b;
    gradient = A.t() * r;
  }

  //! Get the matrix A.
  arma::mat MatrixA() const {return A;}
  //! Modify the matrix A.
  arma::mat& MatrixA() {return A;}

  //! Get the vector b.
  arma::vec Vectorb() const { return b; }
  //! Modify the vector b.
  arma::vec& Vectorb() { return b; }

 private:
  //! Matrix A in square loss function.
  arma::mat A;

  //! Vector b in square loss function.
  arma::vec b;
};

} // namespace ens

#endif
