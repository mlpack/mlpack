
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_nca_test.cpp:

Program Listing for File nca_test.cpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_nca_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/nca_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <mlpack/methods/nca/nca.hpp>
   #include <ensmallen.hpp>
   
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::metric;
   using namespace mlpack::nca;
   using namespace ens;
   
   //
   // Tests for the SoftmaxErrorFunction
   //
   
   TEST_CASE("SoftmaxInitialPoint", "[NCATesT]")
   {
     // Cheap fake dataset.
     arma::mat data;
     data.randu(5, 5);
     arma::Row<size_t> labels;
     labels.zeros(5);
   
     SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);
   
     // Verify the initial point is the identity matrix.
     arma::mat initialPoint = sef.GetInitialPoint();
     for (int row = 0; row < 5; row++)
     {
       for (int col = 0; col < 5; col++)
       {
         if (row == col)
           REQUIRE(initialPoint(row, col) == Approx(1.0).epsilon(1e-7));
         else
           REQUIRE(initialPoint(row, col) == Approx(0.0).margin(1e-5));
       }
     }
   }
   
   /***
    * On a simple fake dataset, ensure that the initial function evaluation is
    * correct.
    */
   TEST_CASE("SoftmaxInitialEvaluation", "[NCATesT]")
   {
     // Useful but simple dataset with six points and two classes.
     arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
     arma::Row<size_t> labels = " 0    0    0    1    1    1   ";
   
     SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);
   
     double objective = sef.Evaluate(arma::eye<arma::mat>(2, 2));
   
     // Result painstakingly calculated by hand by rcurtin (recorded forever in his
     // notebook).  As a result of lack of precision of the by-hand result, the
     // tolerance is fairly high.
     REQUIRE(objective == Approx(-1.5115).epsilon(0.0001));
   }
   
   TEST_CASE("SoftmaxInitialGradient", "[NCATesT]")
   {
     // Useful but simple dataset with six points and two classes.
     arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
     arma::Row<size_t> labels = " 0    0    0    1    1    1   ";
   
     SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);
   
     arma::mat gradient;
     arma::mat coordinates = arma::eye<arma::mat>(2, 2);
     sef.Gradient(coordinates, gradient);
   
     // Results painstakingly calculated by hand by rcurtin (recorded forever in
     // his notebook).  As a result of lack of precision of the by-hand result, the
     // tolerance is fairly high.
     REQUIRE(gradient(0, 0) == Approx(-0.089766).epsilon(0.0005));
     REQUIRE(gradient(1, 0) == Approx(0.0).margin(1e-5));
     REQUIRE(gradient(0, 1) == Approx(0.0).margin(1e-5));
     REQUIRE(gradient(1, 1) == Approx(1.63823).epsilon(0.0001));
   }
   
   TEST_CASE("SoftmaxOptimalEvaluation", "[NCATesT]")
   {
     // Simple optimal dataset.
     arma::mat data           = " 500  500 -500 -500;"
                                "   1    0    1    0 ";
     arma::Row<size_t> labels = "   0    0    1    1 ";
   
     SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);
   
     double objective = sef.Evaluate(arma::eye<arma::mat>(2, 2));
   
     // Use a very close tolerance for optimality; we need to be sure this function
     // gives optimal results correctly.
     REQUIRE(objective == Approx(-4.0).epsilon(1e-12));
   }
   
   TEST_CASE("SoftmaxOptimalGradient", "[NCATesT]")
   {
     // Simple optimal dataset.
     arma::mat data           = " 500  500 -500 -500;"
                                "   1    0    1    0 ";
     arma::Row<size_t> labels = "   0    0    1    1 ";
   
     SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);
   
     arma::mat gradient;
     sef.Gradient(arma::eye<arma::mat>(2, 2), gradient);
   
     REQUIRE(gradient(0, 0) == Approx(0.0).margin(1e-5));
     REQUIRE(gradient(0, 1) == Approx(0.0).margin(1e-5));
     REQUIRE(gradient(1, 0) == Approx(0.0).margin(1e-5));
     REQUIRE(gradient(1, 1) == Approx(0.0).margin(1e-5));
   }
   
   TEST_CASE("SoftmaxSeparableObjective", "[NCATesT]")
   {
     // Useful but simple dataset with six points and two classes.
     arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
     arma::Row<size_t> labels = " 0    0    0    1    1    1   ";
   
     SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);
   
     // Results painstakingly calculated by hand by rcurtin (recorded forever in
     // his notebook).  As a result of lack of precision of the by-hand result, the
     // tolerance is fairly high.
     arma::mat coordinates = arma::eye<arma::mat>(2, 2);
     REQUIRE(sef.Evaluate(coordinates, 0, 1) == Approx(-0.22480).epsilon(0.0001));
     REQUIRE(sef.Evaluate(coordinates, 1, 1) == Approx(-0.30613).epsilon(0.0001));
     REQUIRE(sef.Evaluate(coordinates, 2, 1) == Approx(-0.22480).epsilon(0.0001));
     REQUIRE(sef.Evaluate(coordinates, 3, 1) == Approx(-0.22480).epsilon(0.0001));
     REQUIRE(sef.Evaluate(coordinates, 4, 1) == Approx(-0.30613).epsilon(0.0001));
     REQUIRE(sef.Evaluate(coordinates, 5, 1) == Approx(-0.22480).epsilon(0.0001));
   }
   
   TEST_CASE("OptimalSoftmaxSeparableObjective", "[NCATesT]")
   {
     // Simple optimal dataset.
     arma::mat data           = " 500  500 -500 -500;"
                                "   1    0    1    0 ";
     arma::Row<size_t> labels = "   0    0    1    1 ";
   
     SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);
   
     arma::mat coordinates = arma::eye<arma::mat>(2, 2);
   
     // Use a very close tolerance for optimality; we need to be sure this function
     // gives optimal results correctly.
     REQUIRE(sef.Evaluate(coordinates, 0, 1) == Approx(-1.0).epsilon(1e-12));
     REQUIRE(sef.Evaluate(coordinates, 1, 1) == Approx(-1.0).epsilon(1e-12));
     REQUIRE(sef.Evaluate(coordinates, 2, 1) == Approx(-1.0).epsilon(1e-12));
     REQUIRE(sef.Evaluate(coordinates, 3, 1) == Approx(-1.0).epsilon(1e-12));
   }
   
   TEST_CASE("SoftmaxSeparableGradient", "[NCATesT]")
   {
     // Useful but simple dataset with six points and two classes.
     arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
     arma::Row<size_t> labels = " 0    0    0    1    1    1   ";
   
     SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);
   
     arma::mat coordinates = arma::eye<arma::mat>(2, 2);
     arma::mat gradient(2, 2);
   
     sef.Gradient(coordinates, 0, gradient, 1);
   
     REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.0069708).epsilon(0.0001));
     REQUIRE(gradient(0, 1) == Approx(-2.0 * -0.0101707).epsilon(0.0001));
     REQUIRE(gradient(1, 0) == Approx(-2.0 * -0.0101707).epsilon(0.0001));
     REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.14359).epsilon(0.0001));
   
     sef.Gradient(coordinates, 1, gradient, 1);
   
     REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.008496).epsilon(0.0001));
     REQUIRE(gradient(0, 1) == Approx(0.0).margin(1e-5));
     REQUIRE(gradient(1, 0) == Approx(0.0).margin(1e-5));
     REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.12238).epsilon(0.0001));
   
     sef.Gradient(coordinates, 2, gradient, 1);
   
     REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.0069708).epsilon(0.0001));
     REQUIRE(gradient(0, 1) == Approx(-2.0 * 0.0101707).epsilon(0.0001));
     REQUIRE(gradient(1, 0) == Approx(-2.0 * 0.0101707).epsilon(0.0001));
     REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.1435886).epsilon(0.0001));
   
     sef.Gradient(coordinates, 3, gradient, 1);
   
     REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.0069708).epsilon(0.0001));
     REQUIRE(gradient(0, 1) == Approx(-2.0 * 0.0101707).epsilon(0.0001));
     REQUIRE(gradient(1, 0) == Approx(-2.0 * 0.0101707).epsilon(0.0001));
     REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.1435886).epsilon(0.0001));
   
     sef.Gradient(coordinates, 4, gradient, 1);
   
     REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.008496).epsilon(0.0001));
     REQUIRE(gradient(0, 1) == Approx(0.0).margin(1e-5));
     REQUIRE(gradient(1, 0) == Approx(0.0).margin(1e-5));
     REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.12238).epsilon(0.0001));
   
     sef.Gradient(coordinates, 5, gradient, 1);
   
     REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.0069708).epsilon(0.0001));
     REQUIRE(gradient(0, 1) == Approx(-2.0 * -0.0101707).epsilon(0.0001));
     REQUIRE(gradient(1, 0) == Approx(-2.0 * -0.0101707).epsilon(0.0001));
     REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.1435886).epsilon(0.0001));
   }
   
   //
   // Tests for the NCA algorithm.
   //
   
   TEST_CASE("NCASGDSimpleDataset", "[NCATesT]")
   {
     // Useful but simple dataset with six points and two classes.
     arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
     arma::Row<size_t> labels = " 0    0    0    1    1    1   ";
   
     // Huge learning rate because this is so simple.
     NCA<SquaredEuclideanDistance> nca(data, labels);
     nca.Optimizer().StepSize() = 1.2;
     nca.Optimizer().MaxIterations() = 300000;
     nca.Optimizer().Tolerance() = 0;
     nca.Optimizer().Shuffle() = true;
   
     arma::mat outputMatrix;
     nca.LearnDistance(outputMatrix);
   
     // Ensure that the objective function is better now.
     SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);
   
     double initObj = sef.Evaluate(arma::eye<arma::mat>(2, 2));
     double finalObj = sef.Evaluate(outputMatrix);
     arma::mat finalGradient;
     sef.Gradient(outputMatrix, finalGradient);
   
     // finalObj must be less than initObj.
     REQUIRE(finalObj < initObj);
     // Verify that final objective is optimal.
     REQUIRE(finalObj == Approx(-6.0).epsilon(0.00005));
     // The solution is not unique, so the best we can do is ensure the gradient
     // norm is close to 0.
     REQUIRE(arma::norm(finalGradient, 2) < 1e-4);
   }
   
   TEST_CASE("NCALBFGSSimpleDataset", "[NCATesT]")
   {
     // Useful but simple dataset with six points and two classes.
     arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
     arma::Row<size_t> labels = " 0    0    0    1    1    1   ";
   
     // Huge learning rate because this is so simple.
     NCA<SquaredEuclideanDistance, L_BFGS> nca(data, labels);
     nca.Optimizer().NumBasis() = 5;
   
     arma::mat outputMatrix;
     nca.LearnDistance(outputMatrix);
   
     // Ensure that the objective function is better now.
     SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);
   
     double initObj = sef.Evaluate(arma::eye<arma::mat>(2, 2));
     double finalObj = sef.Evaluate(outputMatrix);
     arma::mat finalGradient;
     sef.Gradient(outputMatrix, finalGradient);
   
     // finalObj must be less than initObj.
     REQUIRE(finalObj < initObj);
     // Verify that final objective is optimal.
     REQUIRE(finalObj == Approx(-6.0).epsilon(1e-7));
     // The solution is not unique, so the best we can do is ensure the gradient
     // norm is close to 0.
     REQUIRE(arma::norm(finalGradient, 2) < 1e-6);
   }
