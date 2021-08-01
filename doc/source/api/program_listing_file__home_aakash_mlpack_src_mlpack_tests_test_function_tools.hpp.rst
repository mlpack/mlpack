
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_test_function_tools.hpp:

Program Listing for File test_function_tools.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_test_function_tools.hpp>` (``/home/aakash/mlpack/src/mlpack/tests/test_function_tools.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_TESTS_TEST_FUNCTION_TOOLS_HPP
   #define MLPACK_TESTS_TEST_FUNCTION_TOOLS_HPP
   
   #include <mlpack/core.hpp>
   
   #include <mlpack/methods/logistic_regression/logistic_regression.hpp>
   
   using namespace mlpack;
   using namespace mlpack::distribution;
   using namespace mlpack::regression;
   
   inline void LogisticRegressionTestData(arma::mat& data,
                                   arma::mat& testData,
                                   arma::mat& shuffledData,
                                   arma::Row<size_t>& responses,
                                   arma::Row<size_t>& testResponses,
                                   arma::Row<size_t>& shuffledResponses)
   {
     // Generate a two-Gaussian dataset.
     GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
     GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));
   
     data = arma::mat(3, 1000);
     responses = arma::Row<size_t>(1000);
     for (size_t i = 0; i < 500; ++i)
     {
       data.col(i) = g1.Random();
       responses[i] = 0;
     }
     for (size_t i = 500; i < 1000; ++i)
     {
       data.col(i) = g2.Random();
       responses[i] = 1;
     }
   
     // Shuffle the dataset.
     arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
         data.n_cols - 1, data.n_cols));
     shuffledData = arma::mat(3, 1000);
     shuffledResponses = arma::Row<size_t>(1000);
     for (size_t i = 0; i < data.n_cols; ++i)
     {
       shuffledData.col(i) = data.col(indices[i]);
       shuffledResponses[i] = responses[indices[i]];
     }
   
     // Create a test set.
     testData = arma::mat(3, 1000);
     testResponses = arma::Row<size_t>(1000);
     for (size_t i = 0; i < 500; ++i)
     {
       testData.col(i) = g1.Random();
       testResponses[i] = 0;
     }
     for (size_t i = 500; i < 1000; ++i)
     {
       testData.col(i) = g2.Random();
       testResponses[i] = 1;
     }
   }
   
   #endif
