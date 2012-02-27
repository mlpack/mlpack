/**
 * @file kernel_pca_test.cpp
 * @author Ajinkya Kale <kaleajinkya@gmail.com>
 *
 * Test file for Kernel PCA.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/methods/kernel_pca/kernel_pca.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(KernelPCATest);

using namespace mlpack;
using namespace mlpack::kpca;
using namespace mlpack::kernel;
using namespace std;
using namespace arma;

BOOST_AUTO_TEST_CASE(linear_kernel)
{

  mat data("1 0 2 3 9;"
            "5 2 8 4 8;"
            "6 7 3 1 8");

  KernelPCA<LinearKernel> p;
  p.Apply(data, 2); // Reduce to 2 dimensions.

  // Compare with correct results.
  mat correct("-1.53781086 -3.51358020 -0.16139887 -1.87706634  7.08985628;"
              " 1.29937798  3.45762685 -2.69910005 -3.15620704  1.09830225");

  // If the eigenvectors are pointed opposite directions, they will cancel
  // each other out in this summation.
  for(size_t i = 0; i < data.n_rows; i++)
  {
    if (fabs(correct(i, 1) + data(i, 1)) < 0.001 /* arbitrary */)
    {
         // Flip eigenvector for this column (negate output).
         data.row(i) *= -1;
    }
  }

  for (size_t row = 0; row < 2; ++row)
    for (size_t col = 0; col < 5; ++col)
      BOOST_REQUIRE_CLOSE(data(row, col), correct(row, col), 1e-3);
}

BOOST_AUTO_TEST_SUITE_END();
