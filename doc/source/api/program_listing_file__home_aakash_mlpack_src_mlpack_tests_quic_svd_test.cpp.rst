
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_quic_svd_test.cpp:

Program Listing for File quic_svd_test.cpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_quic_svd_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/quic_svd_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/quic_svd/quic_svd.hpp>
   
   #include "catch.hpp"
   
   using namespace mlpack;
   
   TEST_CASE("QUICSVDReconstructionError", "[QUICSVDTest]")
   {
     // Load the dataset.
     arma::mat dataset;
     if (!data::Load("test_data_3_1000.csv", dataset))
       FAIL("Cannot load dataset test_data_3_1000.csv");
   
     // The QUIC-SVD procedure can fail---the Monte Carlo error calculation is
     // random.  Therefore we simply require at least one success.
     size_t successes = 0;
     for (size_t i = 0; i < 3; ++i)
     {
       // Obtain the SVD using default parameters.
       arma::mat u, v, sigma;
       svd::QUIC_SVD quicsvd(dataset, u, v, sigma);
   
       // Reconstruct the matrix using the SVD.
       arma::mat reconstruct;
       reconstruct = u * sigma * v.t();
   
       // The relative reconstruction error should be small.
       double relativeError = arma::norm(dataset - reconstruct, "frob") /
                              arma::norm(dataset, "frob");
       if (relativeError < 1e-5)
         ++successes;
     }
   
     REQUIRE(successes > 0);
   }
   
   TEST_CASE("QUICSVDSingularValueError", "[QUICSVDTest]")
   {
     arma::mat U = arma::randn<arma::mat>(3, 20);
     arma::mat V = arma::randn<arma::mat>(10, 3);
   
     arma::mat R;
     arma::qr_econ(U, R, U);
     arma::qr_econ(V, R, V);
   
     arma::mat s = arma::diagmat(arma::vec("1 0.1 0.01"));
   
     arma::mat data = arma::trans(U * arma::diagmat(s) * V.t());
   
     arma::vec s1, s3;
     arma::mat U1, U2, V1, V2, s2;
   
     // Obtain the SVD using default parameters.
     arma::svd_econ(U1, s1, V1, data);
     svd::QUIC_SVD quicsvd(data, U1, V1, s2);
   
     s3 = arma::diagvec(s2);
     s1 = s1.subvec(0, s3.n_elem - 1);
   
     // The sigular value error should be small.
     double error = arma::norm(s1 - s3);
     REQUIRE(error == Approx(0.0).margin(0.1));
   }
   
   TEST_CASE("QUICSVDSameDimensionTest", "[QUICSVDTest]")
   {
     arma::mat dataset = arma::randn<arma::mat>(10, 10);
   
     // Obtain the SVD using default parameters.
     arma::mat u, v, sigma;
     svd::QUIC_SVD quicsvd(dataset, u, v, sigma);
   }
