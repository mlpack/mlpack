
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_softmax_regression_softmax_regression_function.hpp:

Program Listing for File softmax_regression_function.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_softmax_regression_softmax_regression_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/softmax_regression/softmax_regression_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_HPP
   #define MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace regression {
   
   class SoftmaxRegressionFunction
   {
    public:
     SoftmaxRegressionFunction(const arma::mat& data,
                               const arma::Row<size_t>& labels,
                               const size_t numClasses,
                               const double lambda = 0.0001,
                               const bool fitIntercept = false);
   
     const arma::mat InitializeWeights();
   
     void Shuffle();
   
     static const arma::mat InitializeWeights(const size_t featureSize,
                                              const size_t numClasses,
                                              const bool fitIntercept = false);
   
     static void InitializeWeights(arma::mat &weights,
                                   const size_t featureSize,
                                   const size_t numClasses,
                                   const bool fitIntercept = false);
   
     void GetGroundTruthMatrix(const arma::Row<size_t>& labels,
                               arma::sp_mat& groundTruth);
   
     void GetProbabilitiesMatrix(const arma::mat& parameters,
                                 arma::mat& probabilities,
                                 const size_t start,
                                 const size_t batchSize) const;
   
     double Evaluate(const arma::mat& parameters) const;
   
     double Evaluate(const arma::mat& parameters,
                     const size_t start,
                     const size_t batchSize = 1) const;
   
     void Gradient(const arma::mat& parameters, arma::mat& gradient) const;
   
     void Gradient(const arma::mat& parameters,
                   const size_t start,
                   arma::mat& gradient,
                   const size_t batchSize = 1) const;
   
     void PartialGradient(const arma::mat& parameters,
                          size_t j,
                          arma::sp_mat& gradient) const;
   
     const arma::mat& GetInitialPoint() const { return initialPoint; }
   
     size_t NumClasses() const { return numClasses; }
   
     size_t NumFeatures() const
     {
       return initialPoint.n_cols;
     }
     size_t NumFunctions() const { return data.n_cols; }
   
     double& Lambda() { return lambda; }
     double Lambda() const { return lambda; }
   
     bool FitIntercept() const { return fitIntercept; }
   
    private:
     arma::mat data;
     arma::sp_mat groundTruth;
     arma::mat initialPoint;
     size_t numClasses;
     double lambda;
     bool fitIntercept;
   };
   
   } // namespace regression
   } // namespace mlpack
   
   #endif
