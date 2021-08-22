
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
   #include <mlpack/core/data/split_data.hpp>
   
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
   
   template<typename MatType>
   void LoadBostonHousingDataset(MatType& trainData,
                                 MatType& testData,
                                 arma::rowvec& trainResponses,
                                 arma::rowvec& testResponses,
                                 data::DatasetInfo& info)
   {
     MatType dataset;
     arma::rowvec responses;
   
     // Defining categorical deimensions.
     info.SetDimensionality(13);
     info.Type(3) = data::Datatype::categorical;
     info.Type(8) = data::Datatype::categorical;
   
     if (!data::Load("boston_housing_price.csv", dataset, info))
       FAIL("Cannot load test dataset boston_housing_price.csv!");
     if (!data::Load("boston_housing_price_responses.csv", responses))
       FAIL("Cannot load test dataset boston_housing_price_responses.csv!");
   
     data::Split(dataset, responses, trainData, testData,
         trainResponses, testResponses, 0.3);
   }
   
   inline double RMSE(const arma::Row<double>& predictions,
                      const arma::Row<double>& trueResponses)
   {
     double mse = arma::accu(arma::square(predictions - trueResponses)) /
         predictions.n_elem;
     return sqrt(mse);
   }
   
   #endif
