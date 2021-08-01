
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_init_rules_test.cpp:

Program Listing for File init_rules_test.cpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_init_rules_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/init_rules_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/core/math/random.hpp>
   
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <mlpack/methods/ann/ffn.hpp>
   
   #include <mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp>
   #include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
   #include <mlpack/methods/ann/init_rules/oivs_init.hpp>
   #include <mlpack/methods/ann/init_rules/orthogonal_init.hpp>
   #include <mlpack/methods/ann/init_rules/random_init.hpp>
   #include <mlpack/methods/ann/init_rules/const_init.hpp>
   #include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
   #include <mlpack/methods/ann/init_rules/glorot_init.hpp>
   #include <mlpack/methods/ann/init_rules/he_init.hpp>
   #include <mlpack/methods/ann/init_rules/lecun_normal_init.hpp>
   
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::ann;
   
   TEST_CASE("ConstantInitTest", "[InitRulesTest]")
   {
     arma::mat weights;
     RandomInitialization constantInit(1, 1);
     constantInit.Initialize(weights, 100, 100);
   
     bool b = arma::all(arma::vectorise(weights) == 1);
     REQUIRE(b == 1);
   }
   
   TEST_CASE("OrthogonalInitTest", "[InitRulesTest]")
   {
     arma::mat weights;
     OrthogonalInitialization orthogonalInit;
     orthogonalInit.Initialize(weights, 100, 200);
   
     arma::mat orthogonalWeights = arma::eye<arma::mat>(100, 100);
     weights *= weights.t();
   
     for (size_t i = 0; i < weights.n_rows; ++i)
       for (size_t j = 0; j < weights.n_cols; ++j)
       {
         REQUIRE((weights.at(i, j) - orthogonalWeights.at(i, j)) ==
             Approx(0.0).margin(1e-3));
       }
   
     orthogonalInit.Initialize(weights, 200, 100);
     weights = weights.t() * weights;
   
     for (size_t i = 0; i < weights.n_rows; ++i)
       for (size_t j = 0; j < weights.n_cols; ++j)
       {
         REQUIRE((weights.at(i, j) - orthogonalWeights.at(i, j)) ==
             Approx(0.0).margin(1e-3));
       }
   }
   
   TEST_CASE("OrthogonalInitGainTest", "[InitRulesTest]")
   {
     arma::mat weights;
   
     const double gain = 2;
     OrthogonalInitialization orthogonalInit(gain);
     orthogonalInit.Initialize(weights, 100, 200);
   
     arma::mat orthogonalWeights = arma::eye<arma::mat>(100, 100);
     orthogonalWeights *= (gain * gain);
     weights *= weights.t();
   
     for (size_t i = 0; i < weights.n_rows; ++i)
       for (size_t j = 0; j < weights.n_cols; ++j)
       {
         REQUIRE((weights.at(i, j) - orthogonalWeights.at(i, j)) ==
             Approx(0.0).margin(1e-3));
       }
   }
   
   TEST_CASE("ConstInitTest", "[InitRulesTest]")
   {
     arma::mat weights;
     ConstInitialization zeroInit(0);
     zeroInit.Initialize(weights, 100, 100);
   
     bool b = arma::all(arma::vectorise(weights) == 0);
     REQUIRE(b == 1);
   }
   
   /*
    * Simple test of the KathirvalavakumarSubavathiInitialization class with
    * two different sizes.
    */
   TEST_CASE("KathirvalavakumarSubavathiInitTest", "[InitRulesTest]")
   {
     arma::mat data = arma::randu<arma::mat>(100, 1);
   
     arma::mat weights;
     arma::cube weights3d;
   
     KathirvalavakumarSubavathiInitialization kathirvalavakumarSubavathiInit(
         data, 1.5);
   
     kathirvalavakumarSubavathiInit.Initialize(weights, 100, 100);
     kathirvalavakumarSubavathiInit.Initialize(weights3d, 100, 100, 2);
   
     REQUIRE(weights.n_rows == 100);
     REQUIRE(weights.n_cols == 100);
   
     REQUIRE(weights3d.n_rows == 100);
     REQUIRE(weights3d.n_cols == 100);
     REQUIRE(weights3d.n_slices == 2);
   }
   
   TEST_CASE("NguyenWidrowInitTest", "[InitRulesTest]")
   {
     arma::mat weights;
     arma::cube weights3d;
   
     NguyenWidrowInitialization nguyenWidrowInit;
   
     nguyenWidrowInit.Initialize(weights, 100, 100);
     nguyenWidrowInit.Initialize(weights3d, 100, 100, 2);
   
     REQUIRE(weights.n_rows == 100);
     REQUIRE(weights.n_cols == 100);
   
     REQUIRE(weights3d.n_rows == 100);
     REQUIRE(weights3d.n_cols == 100);
     REQUIRE(weights3d.n_slices == 2);
   }
   
   TEST_CASE("OivsInitTest", "[InitRulesTest]")
   {
     arma::mat weights;
     arma::cube weights3d;
   
     OivsInitialization<> oivsInit;
   
     oivsInit.Initialize(weights, 100, 100);
     oivsInit.Initialize(weights3d, 100, 100, 2);
   
     REQUIRE(weights.n_rows == 100);
     REQUIRE(weights.n_cols == 100);
   
     REQUIRE(weights3d.n_rows == 100);
     REQUIRE(weights3d.n_cols == 100);
     REQUIRE(weights3d.n_slices == 2);
   }
   
   TEST_CASE("GaussianInitTest", "[InitRulesTest]")
   {
     const size_t rows = 7;
     const size_t cols = 8;
     const size_t slices = 2;
   
     arma::mat weights;
     arma::cube weights3d;
   
     GaussianInitialization t(0, 0.2);
   
     t.Initialize(weights, rows, cols);
     t.Initialize(weights3d, rows, cols, slices);
   
     REQUIRE(weights.n_rows == rows);
     REQUIRE(weights.n_cols == cols);
   
     REQUIRE(weights3d.n_rows == rows);
     REQUIRE(weights3d.n_cols == cols);
     REQUIRE(weights3d.n_slices == slices);
   }
   
   TEST_CASE("NetworkInitTest", "[InitRulesTest]")
   {
     arma::mat input = arma::ones(5, 1);
     arma::mat response;
     NegativeLogLikelihood<> outputLayer;
   
     // Create a simple network and use the RandomInitialization rule to
     // initialize the network parameters.
     RandomInitialization randomInit(0.5, 0.5);
   
     FFN<NegativeLogLikelihood<>, RandomInitialization> randomModel(
         std::move(outputLayer), randomInit);
     randomModel.Add<IdentityLayer<> >();
     randomModel.Add<Linear<> >(5, 5);
     randomModel.Add<Linear<> >(5, 2);
     randomModel.Add<LogSoftMax<> >();
     randomModel.Predict(input, response);
   
     bool b = arma::all(arma::vectorise(randomModel.Parameters()) == 0.5);
     REQUIRE(b == 1);
     REQUIRE(randomModel.Parameters().n_elem == 42);
   
     // Create a simple network and use the OrthogonalInitialization rule to
     // initialize the network parameters.
     FFN<NegativeLogLikelihood<>, OrthogonalInitialization> orthogonalModel;
     orthogonalModel.Add<IdentityLayer<> >();
     orthogonalModel.Add<Linear<> >(5, 5);
     orthogonalModel.Add<Linear<> >(5, 2);
     orthogonalModel.Add<LogSoftMax<> >();
     orthogonalModel.Predict(input, response);
   
     REQUIRE(orthogonalModel.Parameters().n_elem == 42);
   
     // Create a simple network and use the ZeroInitialization rule to
     // initialize the network parameters.
     FFN<NegativeLogLikelihood<>, ConstInitialization>
       zeroModel(NegativeLogLikelihood<>(), ConstInitialization(0));
     zeroModel.Add<IdentityLayer<> >();
     zeroModel.Add<Linear<> >(5, 5);
     zeroModel.Add<Linear<> >(5, 2);
     zeroModel.Add<LogSoftMax<> >();
     zeroModel.Predict(input, response);
   
     REQUIRE(arma::accu(zeroModel.Parameters()) == 0);
     REQUIRE(zeroModel.Parameters().n_elem == 42);
   
     // Create a simple network and use the
     // KathirvalavakumarSubavathiInitialization rule to initialize the network
     // parameters.
     KathirvalavakumarSubavathiInitialization kathirvalavakumarSubavathiInit(
         input, 1.5);
     FFN<NegativeLogLikelihood<>, KathirvalavakumarSubavathiInitialization>
         ksModel(std::move(outputLayer), kathirvalavakumarSubavathiInit);
     ksModel.Add<IdentityLayer<> >();
     ksModel.Add<Linear<> >(5, 5);
     ksModel.Add<Linear<> >(5, 2);
     ksModel.Add<LogSoftMax<> >();
     ksModel.Predict(input, response);
   
     REQUIRE(ksModel.Parameters().n_elem == 42);
   
     // Create a simple network and use the OivsInitialization rule to
     // initialize the network parameters.
     FFN<NegativeLogLikelihood<>, OivsInitialization<> > oivsModel;
     oivsModel.Add<IdentityLayer<> >();
     oivsModel.Add<Linear<> >(5, 5);
     oivsModel.Add<Linear<> >(5, 2);
     oivsModel.Add<LogSoftMax<> >();
     oivsModel.Predict(input, response);
   
     REQUIRE(oivsModel.Parameters().n_elem == 42);
   
     // Create a simple network and use the GaussianInitialization rule to
     // initialize the network parameters.
     FFN<NegativeLogLikelihood<>, GaussianInitialization> gaussianModel;
     gaussianModel.Add<IdentityLayer<> >();
     gaussianModel.Add<Linear<> >(5, 5);
     gaussianModel.Add<Linear<> >(5, 2);
     gaussianModel.Add<LogSoftMax<> >();
     gaussianModel.Predict(input, response);
   
     REQUIRE(gaussianModel.Parameters().n_elem == 42);
   }
   
   TEST_CASE("GlorotInitUniformTest", "[InitRulesTest]")
   {
     arma::mat weights;
     arma::cube weights3d;
   
     XavierInitialization glorotInit;
   
     glorotInit.Initialize(weights, 100, 100);
     glorotInit.Initialize(weights3d, 100, 100, 2);
   
     REQUIRE(weights.n_rows == 100);
     REQUIRE(weights.n_cols == 100);
   
     REQUIRE(weights3d.n_rows == 100);
     REQUIRE(weights3d.n_cols == 100);
     REQUIRE(weights3d.n_slices == 2);
   }
   
   TEST_CASE("GlorotInitNormalTest", "[InitRulesTest]")
   {
     arma::mat weights;
     arma::cube weights3d;
   
     GlorotInitialization glorotInit;
   
     glorotInit.Initialize(weights, 100, 100);
     glorotInit.Initialize(weights3d, 100, 100, 2);
   
     REQUIRE(weights.n_rows == 100);
     REQUIRE(weights.n_cols == 100);
   
     REQUIRE(weights3d.n_rows == 100);
     REQUIRE(weights3d.n_cols == 100);
     REQUIRE(weights3d.n_slices == 2);
   }
   
   TEST_CASE("HeInitTest", "[InitRulesTest]")
   {
     const size_t rows = 4;
     const size_t cols = 4;
     const size_t slices = 2;
   
     arma::mat weights;
     arma::cube weights3d;
   
     HeInitialization initializer;
   
     initializer.Initialize(weights, rows, cols);
     initializer.Initialize(weights3d, rows, cols, slices);
   
     REQUIRE(weights.n_rows == rows);
     REQUIRE(weights.n_cols == cols);
   
     REQUIRE(weights3d.n_rows == rows);
     REQUIRE(weights3d.n_cols == cols);
     REQUIRE(weights3d.n_slices == slices);
   }
   
   TEST_CASE("LecunNormalInitTest", "[InitRulesTest]")
   {
     const size_t rows = 4;
     const size_t cols = 4;
     const size_t slices = 2;
   
     arma::mat weights;
     arma::cube weights3d;
   
     LecunNormalInitialization initializer;
   
     initializer.Initialize(weights, rows, cols);
     initializer.Initialize(weights3d, rows, cols, slices);
   
     REQUIRE(weights.n_rows == rows);
     REQUIRE(weights.n_cols == cols);
   
     REQUIRE(weights3d.n_rows == rows);
     REQUIRE(weights3d.n_cols == cols);
     REQUIRE(weights3d.n_slices == slices);
   }
