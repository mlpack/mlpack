
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_softmax_regression_softmax_regression.cpp:

Program Listing for File softmax_regression.cpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_softmax_regression_softmax_regression.cpp>` (``/home/aakash/mlpack/src/mlpack/methods/softmax_regression/softmax_regression.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "softmax_regression.hpp"
   
   namespace mlpack {
   namespace regression {
   
   SoftmaxRegression::
   SoftmaxRegression(const size_t inputSize,
                     const size_t numClasses,
                     const bool fitIntercept) :
       numClasses(numClasses),
       lambda(0.0001),
       fitIntercept(fitIntercept)
   {
     SoftmaxRegressionFunction::InitializeWeights(
         parameters, inputSize, numClasses, fitIntercept);
   }
   
   void SoftmaxRegression::Classify(const arma::mat& dataset,
                                    arma::Row<size_t>& labels)
       const
   {
     arma::mat probabilities;
     Classify(dataset, probabilities);
   
     // Prepare necessary data.
     labels.zeros(dataset.n_cols);
     double maxProbability = 0;
   
     // For each test input.
     for (size_t i = 0; i < dataset.n_cols; ++i)
     {
       // For each class.
       for (size_t j = 0; j < numClasses; ++j)
       {
         // If a higher class probability is encountered, change prediction.
         if (probabilities(j, i) > maxProbability)
         {
           maxProbability = probabilities(j, i);
           labels(i) = j;
         }
       }
   
       // Set maximum probability to zero for the next input.
       maxProbability = 0;
     }
   }
   
   void SoftmaxRegression::Classify(const arma::mat& dataset,
                                    arma::Row<size_t>& labels,
                                    arma::mat& probabilities)
       const
   {
     Classify(dataset, probabilities);
   
     // Prepare necessary data.
     labels.zeros(dataset.n_cols);
     double maxProbability = 0;
   
     // For each test input.
     for (size_t i = 0; i < dataset.n_cols; ++i)
     {
       // For each class.
       for (size_t j = 0; j < numClasses; ++j)
       {
         // If a higher class probability is encountered, change prediction.
         if (probabilities(j, i) > maxProbability)
         {
           maxProbability = probabilities(j, i);
           labels(i) = j;
         }
       }
   
       // Set maximum probability to zero for the next input.
       maxProbability = 0;
     }
   }
   
   void SoftmaxRegression::Classify(const arma::mat& dataset,
                                    arma::mat& probabilities)
       const
   {
     util::CheckSameDimensionality(dataset, FeatureSize(),
         "SoftmaxRegression::Classify()");
   
     // Calculate the probabilities for each test input.
     arma::mat hypothesis;
     if (fitIntercept)
     {
       // In order to add the intercept term, we should compute following matrix:
       //     [1; data] = arma::join_cols(ones(1, data.n_cols), data)
       //     hypothesis = arma::exp(parameters * [1; data]).
       //
       // Since the cost of join maybe high due to the copy of original data,
       // split the hypothesis computation to two components.
       hypothesis = arma::exp(
         arma::repmat(parameters.col(0), 1, dataset.n_cols) +
         parameters.cols(1, parameters.n_cols - 1) * dataset);
     }
     else
     {
       hypothesis = arma::exp(parameters * dataset);
     }
   
     probabilities = hypothesis / arma::repmat(arma::sum(hypothesis, 0),
                                               numClasses, 1);
   }
   
   double SoftmaxRegression::ComputeAccuracy(
       const arma::mat& testData,
       const arma::Row<size_t>& labels) const
   {
     arma::Row<size_t> predictions;
   
     // Get predictions for the provided data.
     Classify(testData, predictions);
   
     // Increment count for every correctly predicted label.
     size_t count = 0;
     for (size_t i = 0; i < predictions.n_elem; ++i)
       if (predictions(i) == labels(i))
         count++;
   
     // Return percentage accuracy.
     return (count * 100.0) / predictions.n_elem;
   }
   
   } // namespace regression
   } // namespace mlpack
