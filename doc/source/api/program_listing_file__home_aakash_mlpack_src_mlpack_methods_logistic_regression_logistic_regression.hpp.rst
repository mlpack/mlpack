
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_logistic_regression_logistic_regression.hpp:

Program Listing for File logistic_regression.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_logistic_regression_logistic_regression.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/logistic_regression/logistic_regression.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
   #define MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <ensmallen.hpp>
   
   #include "logistic_regression_function.hpp"
   
   namespace mlpack {
   namespace regression {
   
   template<typename MatType = arma::mat>
   class LogisticRegression
   {
    public:
     LogisticRegression(const MatType& predictors,
                        const arma::Row<size_t>& responses,
                        const double lambda = 0);
   
     LogisticRegression(const MatType& predictors,
                        const arma::Row<size_t>& responses,
                        const arma::rowvec& initialPoint,
                        const double lambda = 0);
   
     LogisticRegression(const size_t dimensionality = 0,
                        const double lambda = 0);
   
     template<typename OptimizerType>
     LogisticRegression(const MatType& predictors,
                        const arma::Row<size_t>& responses,
                        OptimizerType& optimizer,
                        const double lambda);
   
     template<typename OptimizerType = ens::L_BFGS, typename... CallbackTypes>
     double Train(const MatType& predictors,
                  const arma::Row<size_t>& responses,
                  CallbackTypes&&... callbacks);
   
     template<typename OptimizerType, typename... CallbackTypes>
     double Train(const MatType& predictors,
                  const arma::Row<size_t>& responses,
                  OptimizerType& optimizer,
                  CallbackTypes&&... callbacks);
   
     const arma::rowvec& Parameters() const { return parameters; }
     arma::rowvec& Parameters() { return parameters; }
   
     const double& Lambda() const { return lambda; }
     double& Lambda() { return lambda; }
   
     template<typename VecType>
     size_t Classify(const VecType& point,
                     const double decisionBoundary = 0.5) const;
   
     void Classify(const MatType& dataset,
                   arma::Row<size_t>& labels,
                   const double decisionBoundary = 0.5) const;
   
     void Classify(const MatType& dataset,
                   arma::mat& probabilities) const;
   
     double ComputeAccuracy(const MatType& predictors,
                            const arma::Row<size_t>& responses,
                            const double decisionBoundary = 0.5) const;
   
     double ComputeError(const MatType& predictors,
                         const arma::Row<size_t>& responses) const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     arma::rowvec parameters;
     double lambda;
   };
   
   } // namespace regression
   } // namespace mlpack
   
   // Include implementation.
   #include "logistic_regression_impl.hpp"
   
   #endif // MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
