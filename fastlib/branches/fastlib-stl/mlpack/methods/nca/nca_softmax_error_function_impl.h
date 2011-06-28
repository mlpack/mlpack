/***
 * @file nca_softmax_impl.h
 * @author Ryan Curtin
 *
 * Implementation of the Softmax error function.
 */
#ifndef __MLPACK_CORE_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTION_IMPL_H
#define __MLPACK_CORE_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTION_IMPL_H

#include "nca_softmax_error_function.h"

namespace mlpack {
namespace nca {

// Initialize with the given kernel.
template<typename Kernel>
SoftmaxErrorFunction<Kernel>::SoftmaxErrorFunction(const arma::mat& dataset,
                                                   const arma::uvec& labels,
                                                   Kernel kernel) :
  dataset_(dataset), labels_(labels), kernel_(kernel)
{ /* nothing to do */ }

template<typename Kernel>
double SoftmaxErrorFunction<Kernel>::Evaluate(const arma::mat& coordinates) {
  // First stretch the dimensions.
  arma::mat newpoints = coordinates * dataset_;

  // For each point i, we must evaluate the softmax function:
  //   p_ij = exp( -K(x_i, x_j) ) / ( sum_{k != i} ( exp( -K(x_i, x_k) )))
  //   p_i = sum_{j in class of i} p_ij
  // We will do this by keeping track of the denominators for each i as well as
  // the numerators (the sum for all j in class of i).  This will be on the
  // order of O((n * (n + 1)) / 2), which really isn't all that great.
  arma::vec numerators, denominators;
  numerators.zeros(newpoints.n_cols);
  denominators.zeros(newpoints.n_cols);
  for (index_t i = 0; i < newpoints.n_cols; i++) {
    for (index_t j = (i + 1); j < newpoints.n_cols; j++) {
      // Evaluate exp(-K(x_i, x_j)).
      double eval = exp(-kernel_.Evaluate(newpoints.unsafe_col(i),
                                          newpoints.unsafe_col(j)));
      if (eval == 0) {
        IO::Warn << "Kernel evaluation for points " << i << " and " << j <<
            " is 0!" << std::endl;
        IO::Warn << "Value of kernel was " << 
            -kernel_.Evaluate(newpoints.unsafe_col(i), newpoints.unsafe_col(j))
            << std::endl;
      }
      if (eval < 0)
        IO::Warn << "Kernel evaluation for points " << i << " and " << j <<
            " is less than 0!" << std::endl;

      // Add this to the denominators of both i and j: p_ij = p_ji.
      denominators[i] += eval;
      denominators[j] += eval;

      // If i and j are the same class, add to numerator of both.
      if (labels_[i] == labels_[j]) {
        numerators[i] += eval;
        numerators[j] += eval;
      }
    }
  }

  // Now, divide the numerators by the denominators, sum, and return.
  // Our actual objective function is f(A) = sum_i p_i
  numerators /= denominators; // Output is in numerators object.
  
  // Check for bad values.
  for (int i = 0; i < newpoints.n_cols; i++) {
    if (denominators[i] == 0) {
      IO::Warn << "Denominator " << i << " is 0!" << std::endl;
      numerators[i] = 0; // Set correctly.
    }
  }

  return -accu(numerators); // Sum of p_i for all i.  We negate because our
                            // solver minimizes, not maximizes.
};

template<typename Kernel>
void SoftmaxErrorFunction<Kernel>::Gradient(const arma::mat& coordinates,
                                            arma::mat& gradient) {
  // Stretch the dimensions.
  arma::mat newpoints = coordinates * dataset_;

  // The gradient is (see eqn. 5 in the paper, but we have negated the objective
  // function so it changes the sign here too):
  //   -2 * A * sum_i (p_i sum_k (p_ik x_ik x_ik^T) -
  //     sum_{j in class of i} (p_ij x_ij x_ij^T))
  //
  // We first calculate the denominator term for the calculation of p_ij which
  // will come later.  This is O((n * (n + 1)) / 2).
  arma::vec p; // We will store the p_i in here.
  arma::vec denominators;
  p.zeros(newpoints.n_cols);
  denominators.zeros(newpoints.n_cols);

  for (index_t i = 0; i < newpoints.n_cols; i++) {
    for (index_t k = (i + 1); k < newpoints.n_cols; k++) {
      double eval = exp(-kernel_.Evaluate(newpoints.unsafe_col(i),
                                          newpoints.unsafe_col(k)));

      // Add to both possible denominators (for p_i and p_k).
      denominators(i) += eval;
      denominators(k) += eval;
      
      // If i and j are the same class, add to numerator of both.
      if (labels_[i] == labels_[k]) {
        p[i] += eval;
        p[k] += eval;
      }
    }
  }

  // Divide by denominators to finish evaluation of all p_i.
  p /= denominators;
  
  // Check for bad values.
  for (int i = 0; i < newpoints.n_cols; i++) {
    if (denominators[i] == 0) {
      IO::Warn << "Denominator " << i << " is 0!" << std::endl;
      p[i] = 0; // Set correctly.
    }
  }

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
  sum.zeros(newpoints.n_rows, newpoints.n_rows);
  for (index_t i = 0; i < newpoints.n_cols; i++) {
    for (index_t k = 0; k < newpoints.n_cols; k++) {
      // Calculate p_ik and p_ki first.
      double eval = exp(-kernel_.Evaluate(newpoints.unsafe_col(i),
                                          newpoints.unsafe_col(k)));
      double p_ik = 0;
      if (denominators[i] > 0)
        p_ik = eval / denominators(i);

      // Subtract x_i from x_k.  We are not using stretched points here.
      arma::vec x_ik = dataset_.col(i) - dataset_.col(k);
      arma::mat second_term = (x_ik * trans(x_ik));

      arma::mat contribution = p_ik * second_term;

      if (labels_[i] == labels_[k])
        sum += (p[i] - 1) * contribution;
      else
        sum += p[i] * contribution;
    }
  }

  // Assemble the final gradient.
  gradient = -2 * coordinates * sum;
}

template<typename Kernel>
arma::mat SoftmaxErrorFunction<Kernel>::GetInitialPoint() {
  return arma::eye<arma::mat>(dataset_.n_rows, dataset_.n_rows);
}

}; // namespace nca
}; // namespace mlpack

#endif
