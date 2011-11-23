/***
 * @file lmetric.cc
 * @author Ryan Curtin
 *
 * Implementation of template specializations of LMetric class.
 */
#include "lmetric.hpp"

namespace mlpack {
namespace metric {

// L1-metric specializations; the root doesn't matter.
template<>
double LMetric<1, true>::Evaluate(const arma::vec& a, const arma::vec& b) {
  double sum = 0;
  for (size_t i = 0; i < a.n_elem; i++)
    sum += fabs(a[i] - b[i]);

  return sum;
}

template<>
double LMetric<1, false>::Evaluate(const arma::vec& a, const arma::vec& b) {
  double sum = 0;
  for (size_t i = 0; i < a.n_elem; i++)
    sum += fabs(a[i] - b[i]);

  return sum;
}

// L2-metric specializations.
template<>
double LMetric<2, true>::Evaluate(const arma::vec& a, const arma::vec& b) {
  double sum = 0;
  for (size_t i = 0; i < a.n_elem; i++)
    sum += pow(a[i] - b[i], 2.0); // fabs() not necessary when squaring.

  return sqrt(sum);
}

template<>
double LMetric<2, false>::Evaluate(const arma::vec& a, const arma::vec& b) {
  double sum = 0;
  for (size_t i = 0; i < a.n_elem; i++)
    sum += pow(a[i] - b[i], 2.0);

  return sum;
}

// L3-metric specialization (not very likely to be used, but just in case).
template<>
double LMetric<3, true>::Evaluate(const arma::vec& a, const arma::vec& b) {
  double sum = 0;
  for (size_t i = 0; i < a.n_elem; i++)
    sum += pow(fabs(a[i] - b[i]), 3.0);

  return pow(sum, 1.0 / 3.0);
}

template<>
double LMetric<3, false>::Evaluate(const arma::vec& a, const arma::vec& b) {
  double sum = 0;
  for (size_t i = 0; i < a.n_elem; i++)
    sum += pow(fabs(a[i] - b[i]), 3.0);

  return sum;
}

}; // namespace kernel
}; // namespace mlpack
