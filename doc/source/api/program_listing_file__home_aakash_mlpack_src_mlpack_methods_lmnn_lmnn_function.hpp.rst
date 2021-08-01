
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_lmnn_lmnn_function.hpp:

Program Listing for File lmnn_function.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_lmnn_lmnn_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/lmnn/lmnn_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LMNN_FUNCTION_HPP
   #define MLPACK_METHODS_LMNN_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   
   #include "constraints.hpp"
   
   namespace mlpack {
   namespace lmnn {
   
   template<typename MetricType = metric::SquaredEuclideanDistance>
   class LMNNFunction
   {
    public:
     LMNNFunction(const arma::mat& dataset,
                  const arma::Row<size_t>& labels,
                  size_t k,
                  double regularization,
                  size_t range,
                  MetricType metric = MetricType());
   
   
     void Shuffle();
   
     double Evaluate(const arma::mat& transformation);
   
     double Evaluate(const arma::mat& transformation,
                     const size_t begin,
                     const size_t batchSize = 1);
   
     template<typename GradType>
     void Gradient(const arma::mat& transformation, GradType& gradient);
   
     template<typename GradType>
     void Gradient(const arma::mat& transformation,
                   const size_t begin,
                   GradType& gradient,
                   const size_t batchSize = 1);
   
     template<typename GradType>
     double EvaluateWithGradient(const arma::mat& transformation,
                                 GradType& gradient);
   
     template<typename GradType>
     double EvaluateWithGradient(const arma::mat& transformation,
                                 const size_t begin,
                                 GradType& gradient,
                                 const size_t batchSize = 1);
   
     const arma::mat& GetInitialPoint() const { return initialPoint; }
   
     size_t NumFunctions() const { return dataset.n_cols; }
   
     const arma::mat& Dataset() const { return dataset; }
   
     const double& Regularization() const { return regularization; }
     double& Regularization() { return regularization; }
   
     const size_t& K() const { return k; }
     size_t& K() { return k; }
   
     const size_t& Range() const { return range; }
     size_t& Range() { return range; }
   
    private:
     arma::mat dataset;
     arma::Row<size_t> labels;
     arma::mat initialPoint;
     arma::mat transformedDataset;
     arma::Mat<size_t> targetNeighbors;
     arma::Mat<size_t> impostors;
     arma::mat distance;
     size_t k;
     MetricType metric;
     double regularization;
     size_t iteration;
     size_t range;
     Constraints<MetricType> constraint;
     arma::mat pCij;
     arma::vec norm;
     arma::cube evalOld;
     arma::mat maxImpNorm;
     arma::mat transformationOld;
     std::vector<arma::mat> oldTransformationMatrices;
     std::vector<size_t> oldTransformationCounts;
     arma::vec lastTransformationIndices;
     arma::uvec points;
     bool impBounds;
     inline void Precalculate();
     inline void UpdateCache(const arma::mat& transformation,
                             const size_t begin,
                             const size_t batchSize);
     inline void TransDiff(std::map<size_t, double>& transformationDiffs,
                           const arma::mat& transformation,
                           const size_t begin,
                           const size_t batchSize);
   };
   
   } // namespace lmnn
   } // namespace mlpack
   
   #include "lmnn_function_impl.hpp"
   
   #endif
