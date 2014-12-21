/**
 * @file lrsdp_function.hpp
 * @author Ryan Curtin
 * @author Abhishek Laddha
 *
 * A class that represents the objective function which LRSDP optimizes.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_FUNCTION_HPP
#define __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_FUNCTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>

namespace mlpack {
namespace optimization {

/**
 * The objective function that LRSDP is trying to optimize.
 */
class LRSDPFunction
{
 public:
  /**
   * Construct the LRSDPFunction with the given initial point and number of
   * constraints.  Set the A, B, and C matrices for each constraint using the
   * A(), B(), and C() functions.
   */
  LRSDPFunction(const size_t numConstraints,
                const arma::mat& initialPoint);

  /**
   * Evaluate the objective function of the LRSDP (no constraints) at the given
   * coordinates.
   */
  double Evaluate(const arma::mat& coordinates) const;

  /**
   * Evaluate the gradient of the LRSDP (no constraints) at the given
   * coordinates.
   */
  void Gradient(const arma::mat& coordinates, arma::mat& gradient) const;

  /**
   * Evaluate a particular constraint of the LRSDP at the given coordinates.
   */
  double EvaluateConstraint(const size_t index,
                            const arma::mat& coordinates) const;
  /**
   * Evaluate the gradient of a particular constraint of the LRSDP at the given
   * coordinates.
   */
  void GradientConstraint(const size_t index,
                          const arma::mat& coordinates,
                          arma::mat& gradient) const;

  //! Get the number of constraints in the LRSDP.
  size_t NumConstraints() const { return b.n_elem; }

  //! Get the initial point of the LRSDP.
  const arma::mat& GetInitialPoint() const { return initialPoint; }

  //! Return the objective function matrix (C).
  const arma::mat& C() const { return c; }
  //! Modify the objective function matrix (C).
  arma::mat& C() { return c; }

  //! Return the vector of A matrices (which correspond to the constraints).
  const std::vector<arma::mat>& A() const { return a; }
  //! Modify the veector of A matrices (which correspond to the constraints).
  std::vector<arma::mat>& A() { return a; }

  //! Return the vector of modes for the A matrices.
  const arma::uvec& AModes() const { return aModes; }
  //! Modify the vector of modes for the A matrices.
  arma::uvec& AModes() { return aModes; }

  //! Return the vector of B values.
  const arma::vec& B() const { return b; }
  //! Modify the vector of B values.
  arma::vec& B() { return b; }

  //! Return string representation of object.
  std::string ToString() const;

 private:
  //! Objective function matrix c.
  arma::mat c;
  //! A_i for each constraint.
  std::vector<arma::mat> a;
  //! b_i for each constraint.
  arma::vec b;

  //! Initial point.
  arma::mat initialPoint;
  //! 1 if entries in matrix, 0 for normal.
  arma::uvec aModes;
};

};
};

#endif // __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_FUNCTION_HPP
