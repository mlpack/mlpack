
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_logistic_regression_logistic_regression_function.hpp:

Program Listing for File logistic_regression_function.hpp
=========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_logistic_regression_logistic_regression_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/logistic_regression/logistic_regression_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
   #define MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/make_alias.hpp>
   #include <mlpack/core/math/shuffle_data.hpp>
   
   namespace mlpack {
   namespace regression {
   
   template<typename MatType = arma::mat>
   class LogisticRegressionFunction
   {
    public:
     LogisticRegressionFunction(const MatType& predictors,
                                const arma::Row<size_t>& responses,
                                const double lambda = 0);
   
     const double& Lambda() const { return lambda; }
     double& Lambda() { return lambda; }
   
     const MatType& Predictors() const { return predictors; }
     const arma::Row<size_t>& Responses() const { return responses; }
   
     void Shuffle();
   
     double Evaluate(const arma::mat& parameters) const;
   
     double Evaluate(const arma::mat& parameters,
                     const size_t begin,
                     const size_t batchSize = 1) const;
   
     void Gradient(const arma::mat& parameters, arma::mat& gradient) const;
   
     template<typename GradType>
     void Gradient(const arma::mat& parameters,
                   const size_t begin,
                   GradType& gradient,
                   const size_t batchSize = 1) const;
   
     void PartialGradient(const arma::mat& parameters,
                          const size_t j,
                          arma::sp_mat& gradient) const;
   
     template<typename GradType>
     double EvaluateWithGradient(const arma::mat& parameters,
                                 GradType& gradient) const;
   
     template<typename GradType>
     double EvaluateWithGradient(const arma::mat& parameters,
                                 const size_t begin,
                                 GradType& gradient,
                                 const size_t batchSize = 1) const;
   
     size_t NumFunctions() const { return predictors.n_cols; }
   
     size_t NumFeatures() const { return predictors.n_rows + 1; }
   
    private:
     MatType predictors;
     arma::Row<size_t> responses;
     double lambda;
   };
   
   } // namespace regression
   } // namespace mlpack
   
   // Include implementation.
   #include "logistic_regression_function_impl.hpp"
   
   #endif // MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
