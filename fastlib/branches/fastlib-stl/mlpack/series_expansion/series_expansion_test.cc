/***
 * @file series_expansion_test.cc
 * @author Dongryeol Lee, Ryan Curtin
 */
#include <fastlib/fastlib.h>
#include <fastlib/fx/io.h>
#include <complex>
#include "fourier_expansion.h"
#include "fourier_series_expansion_aux.h"
#include "fourier_kernel_aux.h"
#include <mlpack/core/kernels/lmetric.h>

using namespace mlpack;

#define BOOST_TEST_MODULE Series_Expansion_Test
#include <boost/test/unit_test.hpp>

template<typename TKernelAux>
double BaseCase_(const arma::vec& query_point, const arma::mat& reference_set,
                 const arma::vec& reference_weights,
                 const TKernelAux &kernel_aux) {

  double sum = 0.0;

  for (size_t i = 0; i < reference_set.n_cols; i++) {
    double squared_distance = kernel::LMetric<2>::Evaluate(query_point,
        reference_set.unsafe_col(i));
    sum += kernel_aux.kernel_.EvalUnnormOnSq(squared_distance);
  }
  return sum;

}

BOOST_AUTO_TEST_CASE(TestFourierExpansion) {
  GaussianKernelFourierAux<double> kernel_aux;
  size_t order = 3;
  size_t dim = 3;
  double bandwidth = 30;
  kernel_aux.Init(bandwidth, order, dim);

  // Set the integral truncation limit manually.
  kernel_aux.sea_.set_integral_truncation_limit(bandwidth * 3.0);

  // Create an expansion from a random synthetic dataset.
  arma::mat random_dataset(3, 20);
  arma::vec weights(20);
  arma::vec center(3);
  center.zeros();
  for (size_t j = 0; j < 20; j++) {
    for (size_t i = 0; i < 3; i++) {
      random_dataset(i, j) = math::Random(0, i);
    }
  }

  for(size_t i = 0; i < 20; i++)
    center += random_dataset.unsafe_col(i);
  center /= 20.0;

  weights.ones();
  FourierExpansion<GaussianKernelFourierAux<double> > expansion;
  expansion.Init(center, kernel_aux);

  expansion.AccumulateCoeffs(random_dataset, weights, 0, 20, 3);

  // Retrieve the coefficients and print them out.
//  const arma::Col<std::complex<double> > &coeffs = expansion.get_coeffs();

  // Evaluate the expansion, and compare against the naive.
  arma::vec evaluation_point(3);
  evaluation_point.fill(2.0);

  // It is not specified what we should be comparing with, so this is commented
  // out for now.
//    IO::Info << "Expansion evaluated to be: "
//     << expansion.EvaluateField(evaluation_point.memptr(), 3) << std::endl;
//    IO::Info << "Naive sum: " << BaseCase_(evaluation_point, random_dataset,weights, kernel_aux) << std::endl;

}

BOOST_AUTO_TEST_CASE(TestFourierExpansionMapping) {
  FourierSeriesExpansionAux<double> series_aux;
  size_t order = 2;
  size_t dim = 3;
  series_aux.Init(order, dim);

  // Verify that the shifting of each multiindex by the max order
  // roughly corresponds to the base ((2 * order) + 1) number.
  size_t total_num_mapping = (size_t) pow(2 * order + 1, dim);
  for (size_t i = 0; i < total_num_mapping; i++) {
    const std::vector<size_t> &mapping = series_aux.get_multiindex(i);

    size_t number = 0;
//    printf("The mapping: ");
    for  (size_t j = 0; j < dim; j++) {
      number = (2 * order + 1) * number + (mapping[j] + order);
//    printf("%zu ", mapping[j]);
    }

//  printf("maps to %zu\n", number);

    BOOST_REQUIRE_EQUAL(number, i);
  }
}
