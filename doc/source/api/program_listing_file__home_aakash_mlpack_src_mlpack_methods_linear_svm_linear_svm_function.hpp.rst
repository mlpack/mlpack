
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_linear_svm_linear_svm_function.hpp:

Program Listing for File linear_svm_function.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_linear_svm_linear_svm_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/linear_svm/linear_svm_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_HPP
   #define MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace svm {
   
   template <typename MatType = arma::mat>
   class LinearSVMFunction
   {
    public:
     LinearSVMFunction(const MatType& dataset,
                       const arma::Row<size_t>& labels,
                       const size_t numClasses,
                       const double lambda = 0.0001,
                       const double delta = 1.0,
                       const bool fitIntercept = false);
   
     void Shuffle();
   
     static void InitializeWeights(arma::mat& weights,
                                   const size_t featureSize,
                                   const size_t numClasses,
                                   const bool fitIntercept = false);
   
     void GetGroundTruthMatrix(const arma::Row<size_t>& labels,
                               arma::sp_mat& groundTruth);
   
     double Evaluate(const arma::mat& parameters);
   
     double Evaluate(const arma::mat& parameters,
                     const size_t firstId,
                     const size_t batchSize = 1);
   
     template <typename GradType>
     void Gradient(const arma::mat& parameters,
                   GradType& gradient);
   
     template <typename GradType>
     void Gradient(const arma::mat& parameters,
                   const size_t firstId,
                   GradType& gradient,
                   const size_t batchSize = 1);
   
     template <typename GradType>
     double EvaluateWithGradient(const arma::mat& parameters,
                                 GradType& gradient) const;
   
     template <typename GradType>
     double EvaluateWithGradient(const arma::mat& parameters,
                                 const size_t firstId,
                                 GradType& gradient,
                                 const size_t batchSize = 1) const;
   
     const arma::mat& InitialPoint() const { return initialPoint; }
     arma::mat& InitialPoint() { return initialPoint; }
   
     const arma::sp_mat& Dataset() const { return dataset; }
     arma::sp_mat& Dataset() { return dataset; }
   
     double& Lambda() { return lambda; }
     double Lambda() const { return lambda; }
   
     bool FitIntercept() const { return fitIntercept; }
   
     size_t NumFunctions() const;
   
    private:
     arma::mat initialPoint;
   
     arma::sp_mat groundTruth;
   
     MatType dataset;
   
     size_t numClasses;
   
     double lambda;
   
     double delta;
   
     bool fitIntercept;
   };
   
   } // namespace svm
   } // namespace mlpack
   
   // Include implementation
   #include "linear_svm_function_impl.hpp"
   
   #endif // MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_HPP
