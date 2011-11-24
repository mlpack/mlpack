/**
 * @file mvu_objective_function.cpp
 * @author Ryan Curtin
 *
 * Implementation of the MVUObjectiveFunction class.
 */
#include "mvu_objective_function.hpp"

#include <mlpack/neighbor_search/neighbor_search.h>
#include <mlpack/fastica/lin_alg.h>

using namespace mlpack;
using namespace mlpack::mvu;

using mlpack::neighbor::AllkNN;

MVUObjectiveFunction::MVUObjectiveFunction()
{
  // Need to set initial point?  I guess this will be the initial matrix...
  Log::Fatal << "Initialized MVUObjectiveFunction all wrong." << std::endl;
}

MVUObjectiveFunction::MVUObjectiveFunction(arma::mat& initial_point) :
    num_neighbors_(5)
{
  // We will calculate the nearest neighbors of this dataset.
  AllkNN allknn(initial_point);

  allknn.ComputeNeighbors(neighbor_indices_, neighbor_distances_);
//  NOTIFY("Neighbor indices: ");
//  std::cout << neighbor_indices_;
//  NOTIFY("Neighbor distances: ");
//  std::cout << neighbor_distances_;

  // Now shrink the point matrix to the correct target size.
  int dimension = 2; // Get this from CLI: TODO.
  initial_point_ = initial_point;
  initial_point_.shed_rows(2, initial_point.n_rows - 1);

  // Center the matrix, for shits and giggles.
//  arma::mat tmp;
//  linalg__private::Center(initial_point_, tmp);
//  initial_point_ = tmp;
}

double MVUObjectiveFunction::Evaluate(const arma::mat& coordinates)
{
  // We replaced the SDP constraint (K > 0) with (K = R^T R) so now our
  // objective function is simply that (and R is our coordinate matrix).  Since
  // the problem is a maximization problem, we simply negate the objective
  // function and it becomes a minimization problem (which is what L-BFGS and
  // AugLagrangian use).
  double objective = 0;

  for (int i = 0; i < coordinates.n_cols; i++)
    objective -= dot(coordinates.unsafe_col(i), coordinates.unsafe_col(i));

  return objective;
}

void MVUObjectiveFunction::Gradient(const arma::mat& coordinates,
                                    arma::mat& gradient)
{
  // Our objective, f(R) = sum_{ij} (R^T R)_ij, is differentiable into
  //   f'(R) = 2 * R.
  Log::Info << "Coordinates are:\n" << coordinates;

  gradient = 2 * coordinates;

  Log::Info << "Calculated gradient is\n" << gradient;

  std::cout << gradient;
}

double MVUObjectiveFunction::EvaluateConstraint(int index,
                                                const arma::mat& coordinates)
{
  if (index == 0)
  {
    // We are considering the first constraint:
    //   sum (R^T * R) = 0

    // This is a naive implementation; we may be able to improve upon it
    // significantly by avoiding the actual calculation of the Gram matrix
    // (R^T * R).
    if (accu(trans(coordinates) * coordinates) > 0)
    {
      Log::Debug << "Constraint 0 is nonzero: " <<
          accu(trans(coordinates) * coordinates) << std::endl;
    }

    return accu(trans(coordinates) * coordinates);
  }

  // Return 0 for any constraints which are out of bounds.
  if (index >= NumConstraints() || index < 0)
    return 0;

  // Any other constraints are the individual nearest neighbor constraints:
  //   (R^T R)_ii - 2 (R^T R)_ij + (R^T R)_jj - || x_i - x_j ||^2 = 0
  //
  // We will get the i and j values from the given index.
  int i = floor(((double) (index - 1)) / (double) num_neighbors_);
  int j = neighbor_indices_[index - 1]; // Retrieve index of this neighbor.

  // (R^T R)_ij = R.col(i) * R.col(j)  (dot product)
  double rrt_ii = dot(coordinates.col(i), coordinates.col(i));
  double rrt_ij = dot(coordinates.col(i), coordinates.col(j));
  double rrt_jj = dot(coordinates.col(j), coordinates.col(j));

  // We must remember the actual distance between points.
//  NOTIFY("Index %d: i is %d, j is %d", index, i, j);
//  NOTIFY("Neighbor %d: distance of %lf", index - 1,
//      neighbor_distances_[index - 1]);
//  NOTIFY("rrt_ii: %lf; rrt_ij: %lf; rrt_jj: %lf", rrt_ii, rrt_ij, rrt_jj);
//  NOTIFY("r_i: ");
//  std::cout << coordinates.col(i);
//  NOTIFY("r_j: ");
//  std::cout << coordinates.col(j);
//  NOTIFY("LHS: %lf", (rrt_ii - 2 * rrt_ij + rrt_jj));

  if (((rrt_ii - 2 * rrt_ij + rrt_jj) -
      neighbor_distances_[index - 1]) > 1e-5)
  {
    Log::Debug << "Constraint " << index << " is nonzero: " <<
        ((rrt_ii - 2 * rrt_ij + rrt_jj) - neighbor_distances_[index - 1])
        << std::endl;
  }

  return ((rrt_ii - 2 * rrt_ij + rrt_jj) - neighbor_distances_[index - 1]);
}

void MVUObjectiveFunction::GradientConstraint(int index,
                                              const arma::mat& coordinates,
                                              arma::mat& gradient)
{
  // Set gradient to 0 (we will add to it).
  gradient.zeros(coordinates.n_rows, coordinates.n_cols);

  // Return 0 for any constraints which are out of bounds.
  if (index >= NumConstraints() || index < 0)
    return;

  if (index == 0)
  {
    // We consider the gradient of the first constraint:
    //   sum (R^T * R) = 0
    // It is eventually worked out that
    //   d / dR_xy (sum (R^T R)) = sum_i (R_xi)
    //
    // We can see that we can separate this out into two distinct sums, for each
    // row and column, so we can loop first over the columns and then over the
    // rows to assemble the entire gradient matrix.
    //for (int i = 0; i < coordinates.n_cols; i++)
    //  gradient.col(i) += accu(coordinates.col(i)); // sum_i (R_xi)
    arma::mat ones(gradient.n_cols, gradient.n_cols);
    gradient = coordinates * ones;

    return;
  }

  // Any other constraints are the individual nearest neighbor constraints:
  //  (R^T R)_ii - 2 (R^T R)_ij + (R^T R)_jj = || x_i - x_j ||^2
  //
  // We will get the i and j values from the given index.
  int i = floor(((double) (index - 1)) / (double) num_neighbors_);
  int j = neighbor_indices_[index - 1];

  // The gradient matrix for the nearest neighbor constraint (i, j) is zero
  // except for column i, which is equal to 2 (R.col(i) - R.col(j)) and also
  // except for column j, which is equal to 2 (R.col(j) - R.col(i)).
  arma::vec diff_row = coordinates.col(i) - coordinates.col(j);
  gradient.col(i) += 2 * diff_row;
  gradient.col(j) -= 2 * diff_row;
}
