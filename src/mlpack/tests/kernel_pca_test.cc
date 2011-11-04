/*
 * =====================================================================================
 *
 *       Filename:  kernel_pca_test.cc
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  01/09/2008 11:26:48 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include <mlpack/methods/kernel_pca/kernel_pca.h>
#include <vector>
#include <mlpack/core.h>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(KernelPCATest);

  BOOST_AUTO_TEST_CASE(TestGeneralKernelPCA) {
     arma::mat eigen_vectors;
     arma::vec eigen_values;

     KernelPCA engine_;
     KernelPCA::GaussianKernel kernel_;
     engine_.Init("test_data_3_1000.csv", 5, 20);
     engine_.ComputeNeighborhoods();
     double bandwidth;
     engine_.EstimateBandwidth(&bandwidth);
     kernel_.set(bandwidth);
     engine_.LoadAffinityMatrix();
     engine_.ComputeGeneralKernelPCA(kernel_, 15,
                                      &eigen_vectors,
                                      &eigen_values);
     engine_.SaveToTextFile("results", eigen_vectors, eigen_values);
    }

  BOOST_AUTO_TEST_CASE(TestLLE) {
      arma::mat eigen_vectors;
      arma::vec eigen_values;
      KernelPCA engine_;
      KernelPCA::GaussianKernel kernel_;
      engine_.Init("test_data_3_1000.csv", 5, 20);
      engine_.ComputeNeighborhoods();
      engine_.LoadAffinityMatrix();
      engine_.ComputeLLE(2,
                           &eigen_vectors,
                           &eigen_values);
      engine_.SaveToTextFile("results", eigen_vectors, eigen_values);
  }

  BOOST_AUTO_TEST_CASE (TestSpectralRegression) {
      KernelPCA engine_;
      KernelPCA::GaussianKernel kernel_;
      engine_.Init("test_data_3_1000.csv", 5, 20);
      engine_.ComputeNeighborhoods();
      double bandwidth;
      engine_.EstimateBandwidth(&bandwidth);
      kernel_.set(bandwidth);
      engine_.LoadAffinityMatrix();
      std::map<size_t, size_t> data_label;
      for(size_t i=0; i<20; i++) {
        data_label[math::RandInt(0, engine_->data_.n_cols())] =
          math::RandInt(0 ,2);
      }
      arma::mat embedded_coordinates;
      arma::vec eigenvalues;
      engine_.ComputeSpectralRegression(kernel_,
                                         data_label,
                                         &embedded_coordinates,
                                         &eigenvalues);
      engine_.SaveToTextFile("results", embedded_coordinates, eigenvalues);
  }

BOOST_AUTO_TEST_SUITE_END();
