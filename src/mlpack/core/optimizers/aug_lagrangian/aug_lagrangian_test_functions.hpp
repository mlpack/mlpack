/**
 * @file aug_lagrangian_test_functions.hpp
 * @author Ryan Curtin
 *
 * Define test functions for the augmented Lagrangian method.
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
#ifndef __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_TEST_FUNCTIONS_HPP
#define __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_TEST_FUNCTIONS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

/**
 * This function is taken from "Practical Mathematical Optimization" (Snyman),
 * section 5.3.8 ("Application of the Augmented Lagrangian Method").  It has
 * only one constraint.
 *
 * The minimum that satisfies the constraint is x = [1, 4], with an objective
 * value of 70.
 */
class AugLagrangianTestFunction
{
 public:
  AugLagrangianTestFunction();
  AugLagrangianTestFunction(const arma::mat& initial_point);

  double Evaluate(const arma::mat& coordinates);
  void Gradient(const arma::mat& coordinates, arma::mat& gradient);

  size_t NumConstraints() const { return 1; }

  double EvaluateConstraint(const size_t index, const arma::mat& coordinates);
  void GradientConstraint(const size_t index,
                          const arma::mat& coordinates,
                          arma::mat& gradient);

  const arma::mat& GetInitialPoint() const { return initialPoint; }

  // convert the obkect into a string
  std::string ToString() const;

 private:
  arma::mat initialPoint;
};

/**
 * This function is taken from M. Gockenbach's lectures on general nonlinear
 * programs, found at:
 * http://www.math.mtu.edu/~msgocken/ma5630spring2003/lectures/nlp/nlp.pdf
 *
 * The program we are using is example 2.5 from this document.
 * I have arbitrarily decided that this will be called the Gockenbach function.
 *
 * The minimum that satisfies the two constraints is given as
 *   x = [0.12288, -1.1078, 0.015100], with an objective value of about 29.634.
 */
class GockenbachFunction
{
 public:
  GockenbachFunction();
  GockenbachFunction(const arma::mat& initial_point);

  double Evaluate(const arma::mat& coordinates);
  void Gradient(const arma::mat& coordinates, arma::mat& gradient);

  size_t NumConstraints() const { return 2; };

  double EvaluateConstraint(const size_t index, const arma::mat& coordinates);
  void GradientConstraint(const size_t index,
                          const arma::mat& coordinates,
                          arma::mat& gradient);

  const arma::mat& GetInitialPoint() const { return initialPoint; }

 private:
  arma::mat initialPoint;
};



/**
 * This function is the Lovasz-Theta semidefinite program, as implemented in the
 * following paper:
 *
 * S. Burer, R. Monteiro
 * "A nonlinear programming algorithm for solving semidefinite programs via
 * low-rank factorization."
 * Journal of Mathematical Programming, 2004
 *
 * Given a simple, undirected graph G = (V, E), the Lovasz-Theta SDP is defined
 * by:
 *
 * min_X{Tr(-(e e^T)^T X) : Tr(X) = 1, X_ij = 0 for all (i, j) in E, X >= 0}
 *
 * where e is the vector of all ones and X has dimension |V| x |V|.
 *
 * In the Monteiro-Burer formulation, we take X = R * R^T, where R is the
 * coordinates given to the Evaluate(), Gradient(), EvaluateConstraint(), and
 * GradientConstraint() functions.
 */
class LovaszThetaSDP
{
 public:
  LovaszThetaSDP();

  /**
   * Initialize the Lovasz-Theta SDP with the given set of edges.  The edge
   * matrix should consist of rows of two dimensions, where dimension 0 is the
   * first vertex of the edge and dimension 1 is the second edge (or vice versa,
   * as it doesn't make a difference).
   *
   * @param edges Matrix of edges.
   */
  LovaszThetaSDP(const arma::mat& edges);

  double Evaluate(const arma::mat& coordinates);
  void Gradient(const arma::mat& coordinates, arma::mat& gradient);

  size_t NumConstraints() const;

  double EvaluateConstraint(const size_t index, const arma::mat& coordinates);
  void GradientConstraint(const size_t index,
                          const arma::mat& coordinates,
                          arma::mat& gradient);

  const arma::mat& GetInitialPoint();

  const arma::mat& Edges() const { return edges; }
  arma::mat&       Edges()       { return edges; }

 private:
  arma::mat edges;
  size_t vertices;

  arma::mat initialPoint;
};

}; // namespace optimization
}; // namespace mlpack

#endif // __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_TEST_FUNCTIONS_HPP
