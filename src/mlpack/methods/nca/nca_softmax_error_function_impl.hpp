/**
 * @file nca_softmax_impl.h
 * @author Ryan Curtin
 *
 * Implementation of the Softmax error function.
 * This file is part of MLPACK 1.0.2.
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
#ifndef __MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTCLIN_IMPL_H
#define __MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTCLIN_IMPL_H

// In case it hasn't been included already.
#include "nca_softmax_error_function.hpp"

namespace mlpack {
namespace nca {

// Initialize with the given kernel.
template<typename Kernel>
SoftmaxErrorFunction<Kernel>::SoftmaxErrorFunction(const arma::mat& dataset,
                                                   const arma::uvec& labels,
                                                   Kernel kernel) :
  dataset_(dataset), labels_(labels), kernel_(kernel),
  last_coordinates_(dataset.n_rows, dataset.n_rows),
  precalculated_(false)
{ /* nothing to do */ }

template<typename Kernel>
double SoftmaxErrorFunction<Kernel>::Evaluate(const arma::mat& coordinates)
{
  // Calculate the denominators and numerators, if necessary.
  Precalculate(coordinates);

  return -accu(p_); // Sum of p_i for all i.  We negate because our solver
                    // minimizes, not maximizes.
};

template<typename Kernel>
void SoftmaxErrorFunction<Kernel>::Gradient(const arma::mat& coordinates,
                                            arma::mat& gradient)
{
  // Calculate the denominators and numerators, if necessary.
  Precalculate(coordinates);

  // Now, we handle the summation over i:
  //   sum_i (p_i sum_k (p_ik x_ik x_ik^T) -
  //       sum_{j in class of i} (p_ij x_ij x_ij^T)
  // We can algebraically manipulate the whole thing to produce a more
  // memory-friendly way to calculate this.  Looping over each i and k (again
  // O((n * (n + 1)) / 2) as with the last step, we can add the following to the
  // sum:
  //
  //   if class of i is the same as the class of k, add
  //     (((p_i - (1 / p_i)) p_ik) + ((p_k - (1 / p_k)) p_ki)) x_ik x_ik^T
  //   otherwise, add
  //     (p_i p_ik + p_k p_ki) x_ik x_ik^T
  arma::mat sum;
  sum.zeros(stretched_dataset_.n_rows, stretched_dataset_.n_rows);
  for (size_t i = 0; i < stretched_dataset_.n_cols; i++)
  {
    for (size_t k = (i + 1); k < stretched_dataset_.n_cols; k++)
    {
      // Calculate p_ik and p_ki first.
      double eval = exp(-kernel_.Evaluate(stretched_dataset_.unsafe_col(i),
                                          stretched_dataset_.unsafe_col(k)));
      double p_ik = 0, p_ki = 0;
      p_ik = eval / denominators_(i);
      p_ki = eval / denominators_(k);

      // Subtract x_i from x_k.  We are not using stretched points here.
      arma::vec x_ik = dataset_.col(i) - dataset_.col(k);
      arma::mat second_term = (x_ik * trans(x_ik));

      if (labels_[i] == labels_[k])
        sum += ((p_[i] - 1) * p_ik + (p_[k] - 1) * p_ki) * second_term;
      else
        sum += (p_[i] * p_ik + p_[k] * p_ki) * second_term;
    }
  }

  // Assemble the final gradient.
  gradient = -2 * coordinates * sum;
}

template<typename Kernel>
const arma::mat SoftmaxErrorFunction<Kernel>::GetInitialPoint() const
{
  return arma::eye<arma::mat>(dataset_.n_rows, dataset_.n_rows);
}

template<typename Kernel>
void SoftmaxErrorFunction<Kernel>::Precalculate(const arma::mat& coordinates)
{
  // Make sure the calculation is necessary.
  if ((accu(coordinates == last_coordinates_) == coordinates.n_elem) &&
      precalculated_)
    return; // No need to calculate; we already have this stuff saved.

  // Coordinates are different; save the new ones, and stretch the dataset.
  last_coordinates_ = coordinates;
  stretched_dataset_ = coordinates * dataset_;

  // For each point i, we must evaluate the softmax function:
  //   p_ij = exp( -K(x_i, x_j) ) / ( sum_{k != i} ( exp( -K(x_i, x_k) )))
  //   p_i = sum_{j in class of i} p_ij
  // We will do this by keeping track of the denominators for each i as well as
  // the numerators (the sum for all j in class of i).  This will be on the
  // order of O((n * (n + 1)) / 2), which really isn't all that great.
  p_.zeros(stretched_dataset_.n_cols);
  denominators_.zeros(stretched_dataset_.n_cols);
  for (size_t i = 0; i < stretched_dataset_.n_cols; i++)
  {
    for (size_t j = (i + 1); j < stretched_dataset_.n_cols; j++)
    {
      // Evaluate exp(-K(x_i, x_j)).
      double eval = exp(-kernel_.Evaluate(stretched_dataset_.unsafe_col(i),
                                          stretched_dataset_.unsafe_col(j)));

      // Add this to the denominators of both i and j: p_ij = p_ji.
      denominators_[i] += eval;
      denominators_[j] += eval;

      // If i and j are the same class, add to numerator of both.
      if (labels_[i] == labels_[j])
      {
        p_[i] += eval;
        p_[j] += eval;
      }
    }
  }

  // Divide p_i by their denominators.
  p_ /= denominators_;

  // Clean up any bad values.
  for (size_t i = 0; i < stretched_dataset_.n_cols; i++)
  {
    if (denominators_[i] == 0.0)
    {
      Log::Debug << "Denominator of p_{" << i << ", j} is 0." << std::endl;

      // Set to usable values.
      denominators_[i] = std::numeric_limits<double>::infinity();
      p_[i] = 0;
    }
  }

  // We've done a precalculation.  Mark it as done.
  precalculated_ = true;
}

}; // namespace nca
}; // namespace mlpack

#endif
