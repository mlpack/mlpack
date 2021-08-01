
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_scaling_test.cpp:

Program Listing for File scaling_test.cpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_scaling_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/scaling_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/core/data/scaler_methods/pca_whitening.hpp>
   #include <mlpack/core/data/scaler_methods/zca_whitening.hpp>
   #include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
   #include <mlpack/core/data/scaler_methods/max_abs_scaler.hpp>
   #include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
   #include <mlpack/core/data/scaler_methods/mean_normalization.hpp>
   
   #include "test_catch_tools.hpp"
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::data;
   using namespace std;
   
   arma::mat dataset = "-1 -0.5 0 1;"
                       "2 6 10 18;";
   arma::mat scaleddataset;
   arma::mat temp;
   
   TEST_CASE("MinMaxScalerTest", "[ScalingTest]")
   {
     arma::mat scaled = "0 0.2500 0.5000 1.000;"
                        "0 0.2500 0.5000 1.000;";
     data::MinMaxScaler scale;
     scale.Fit(dataset);
     scale.Transform(dataset, scaleddataset);
     scale.InverseTransform(scaleddataset, temp);
     CheckMatrices(scaleddataset, scaled);
     CheckMatrices(dataset, temp);
   }
   
   TEST_CASE("MaxAbsScalerTest", "[ScalingTest]")
   {
     arma::mat scaled = "-1 -0.5 0 1;"
                        "0.1111111111 0.3333333333 0.55555556 1.0000;";
     data::MaxAbsScaler scale;
     scale.Fit(dataset);
     scale.Transform(dataset, scaleddataset);
     scale.InverseTransform(scaleddataset, temp);
     CheckMatrices(scaleddataset, scaled);
     CheckMatrices(dataset, temp);
   }
   
   TEST_CASE("StandardScalerTest", "[ScalingTest]")
   {
     arma::mat scaled = "-1.18321596 -0.50709255  0.16903085 1.52127766;"
                        "-1.18321596 -0.50709255  0.16903085 1.52127766;";
     data::StandardScaler scale;
     scale.Fit(dataset);
     scale.Transform(dataset, scaleddataset);
     scale.InverseTransform(scaleddataset, temp);
     CheckMatrices(scaleddataset, scaled);
     CheckMatrices(dataset, temp);
   }
   
   TEST_CASE("MeanNormalizationTest", "[ScalingTest]")
   {
     arma::mat scaled = "-0.43750000000 -0.187500000 0.062500000 0.562500000;"
                        "-0.43750000000 -0.187500000 0.062500000 0.562500000;";
     data::MeanNormalization scale;
     scale.Fit(dataset);
     scale.Transform(dataset, scaleddataset);
     scale.InverseTransform(scaleddataset, temp);
     CheckMatrices(scaleddataset, scaled);
     CheckMatrices(dataset, temp);
   }
   
   TEST_CASE("SameInputOutputTest", "[ScalingTest]")
   {
     temp = dataset;
     arma::mat scaled = "-0.43750000000 -0.187500000 0.062500000 0.562500000;"
                        "-0.43750000000 -0.187500000 0.062500000 0.562500000;";
     data::MeanNormalization scale;
     scale.Fit(temp);
     scale.Transform(temp, temp);
     CheckMatrices(scaled, temp);
     scale.InverseTransform(temp, temp);
     CheckMatrices(dataset, temp);
   }
   
   TEST_CASE("ZeroMatrixTest", "[ScalingTest]")
   {
     arma::mat input(2, 4, arma::fill::zeros);
     data::MeanNormalization scale;
     scale.Fit(input);
     scale.Transform(input, temp);
     CheckMatrices(input, temp);
     scale.InverseTransform(input, temp);
     CheckMatrices(input, temp);
   }
   
   TEST_CASE("ZeroScaleTest", "[ScalingTest]")
   {
     dataset = "1 1 1 1;"
               "2 6 10 18;";
     arma::mat scaled = "0 0 0 0;"
                        "0 0.2500 0.5000 1.000;";
     data::MinMaxScaler scale;
     scale.Fit(dataset);
     scale.Transform(dataset, scaleddataset);
     scale.InverseTransform(scaleddataset, temp);
     CheckMatrices(scaleddataset, scaled);
     CheckMatrices(dataset, temp);
   }
   
   TEST_CASE("PCAWhiteningTest", "[ScalingTest]")
   {
     data::PCAWhitening scale;
     arma::mat output;
     scale.Fit(dataset);
     scale.Transform(dataset, output);
     arma::vec diagonals = (mlpack::math::ColumnCovariance(output)).diag();
     // Checking covarience is close to 1.0
     double ccovsum = 0.0;
     for (size_t i = 0; i < diagonals.n_elem; ++i)
       ccovsum += diagonals(i);
     REQUIRE(ccovsum == Approx(1.0).epsilon(1e-5));
     scale.InverseTransform(output, temp);
     CheckMatrices(dataset, temp);
   }
   
   TEST_CASE("ZCAWhiteningTest", "[ScalingTest]")
   {
     data::ZCAWhitening scale;
     arma::mat output;
     scale.Fit(dataset);
     scale.Transform(dataset, output);
     arma::vec diagonals = (mlpack::math::ColumnCovariance(output)).diag();
     // Check that the covariance is close to 1.0.
     double ccovsum = 0.0;
     for (size_t i = 0; i < diagonals.n_elem; ++i)
       ccovsum += diagonals(i);
     REQUIRE(ccovsum == Approx(1.0).epsilon(1e-5));
     scale.InverseTransform(output, temp);
     CheckMatrices(dataset, temp);
   }
