
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_softmax_regression_softmax_regression.hpp:

Program Listing for File softmax_regression.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_softmax_regression_softmax_regression.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/softmax_regression/softmax_regression.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_HPP
   #define MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <ensmallen.hpp>
   
   #include "softmax_regression_function.hpp"
   
   namespace mlpack {
   namespace regression {
   
   class SoftmaxRegression
   {
    public:
     SoftmaxRegression(const size_t inputSize = 0,
                       const size_t numClasses = 0,
                       const bool fitIntercept = false);
     template<typename OptimizerType = ens::L_BFGS>
     SoftmaxRegression(const arma::mat& data,
                       const arma::Row<size_t>& labels,
                       const size_t numClasses,
                       const double lambda = 0.0001,
                       const bool fitIntercept = false,
                       OptimizerType optimizer = OptimizerType());
     template<typename OptimizerType, typename... CallbackTypes>
     SoftmaxRegression(const arma::mat& data,
                       const arma::Row<size_t>& labels,
                       const size_t numClasses,
                       const double lambda,
                       const bool fitIntercept,
                       OptimizerType optimizer,
                       CallbackTypes&&... callbacks);
     void Classify(const arma::mat& dataset, arma::Row<size_t>& labels) const;
     template<typename VecType>
     size_t Classify(const VecType& point) const;
   
     void Classify(const arma::mat& dataset,
                   arma::Row<size_t>& labels,
                   arma::mat& probabilities) const;
   
     void Classify(const arma::mat& dataset,
                   arma::mat& probabilities) const;
   
     double ComputeAccuracy(const arma::mat& testData,
                            const arma::Row<size_t>& labels) const;
     template<typename OptimizerType = ens::L_BFGS>
     double Train(const arma::mat& data,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  OptimizerType optimizer = OptimizerType());
     template<typename OptimizerType = ens::L_BFGS, typename... CallbackTypes>
     double Train(const arma::mat& data,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  OptimizerType optimizer,
                  CallbackTypes&&... callbacks);
   
     size_t& NumClasses() { return numClasses; }
     size_t NumClasses() const { return numClasses; }
   
     double& Lambda() { return lambda; }
     double Lambda() const { return lambda; }
   
     bool FitIntercept() const { return fitIntercept; }
   
     arma::mat& Parameters() { return parameters; }
     const arma::mat& Parameters() const { return parameters; }
   
     size_t FeatureSize() const
     { return fitIntercept ? parameters.n_cols - 1:
                             parameters.n_cols; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(parameters));
       ar(CEREAL_NVP(numClasses));
       ar(CEREAL_NVP(lambda));
       ar(CEREAL_NVP(fitIntercept));
     }
   
    private:
     arma::mat parameters;
     size_t numClasses;
     double lambda;
     bool fitIntercept;
   };
   
   } // namespace regression
   } // namespace mlpack
   
   // Include implementation.
   #include "softmax_regression_impl.hpp"
   
   #endif
