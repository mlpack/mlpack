
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_linear_svm_linear_svm.hpp:

Program Listing for File linear_svm.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_linear_svm_linear_svm.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/linear_svm/linear_svm.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_HPP
   #define MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <ensmallen.hpp>
   
   #include "linear_svm_function.hpp"
   
   namespace mlpack {
   namespace svm {
   
   template <typename MatType = arma::mat>
   class LinearSVM
   {
    public:
     template <typename OptimizerType, typename... CallbackTypes>
     LinearSVM(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const double lambda,
               const double delta,
               const bool fitIntercept,
               OptimizerType optimizer,
               CallbackTypes&&... callbacks);
   
     template <typename OptimizerType = ens::L_BFGS>
     LinearSVM(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses = 2,
               const double lambda = 0.0001,
               const double delta = 1.0,
               const bool fitIntercept = false,
               OptimizerType optimizer = OptimizerType());
   
     LinearSVM(const size_t inputSize,
               const size_t numClasses = 0,
               const double lambda = 0.0001,
               const double delta = 1.0,
               const bool fitIntercept = false);
     LinearSVM(const size_t numClasses = 0,
               const double lambda = 0.0001,
               const double delta = 1.0,
               const bool fitIntercept = false);
   
     void Classify(const MatType& data,
                   arma::Row<size_t>& labels) const;
   
     void Classify(const MatType& data,
                   arma::Row<size_t>& labels,
                   arma::mat& scores) const;
   
     void Classify(const MatType& data,
                   arma::mat& scores) const;
   
     template<typename VecType>
     size_t Classify(const VecType& point) const;
   
     double ComputeAccuracy(const MatType& testData,
                            const arma::Row<size_t>& testLabels) const;
   
     template <typename OptimizerType, typename... CallbackTypes>
     double Train(const MatType& data,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  OptimizerType optimizer,
                  CallbackTypes&&... callbacks);
   
     template <typename OptimizerType = ens::L_BFGS>
     double Train(const MatType& data,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses = 2,
                  OptimizerType optimizer = OptimizerType());
   
   
     size_t& NumClasses() { return numClasses; }
     size_t NumClasses() const { return numClasses; }
   
     double& Lambda() { return lambda; }
     double Lambda() const { return lambda; }
   
     double& Delta() { return delta; }
     double Delta() const { return delta; }
   
     bool& FitIntercept() { return fitIntercept; }
   
     arma::mat& Parameters() { return parameters; }
     const arma::mat& Parameters() const { return parameters; }
   
     size_t FeatureSize() const
     { return fitIntercept ? parameters.n_rows - 1 :
              parameters.n_rows; }
   
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
     double delta;
     bool fitIntercept;
   };
   
   } // namespace svm
   } // namespace mlpack
   
   // Include implementation.
   #include "linear_svm_impl.hpp"
   
   #endif // MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_HPP
