
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_nca_nca_softmax_error_function.hpp:

Program Listing for File nca_softmax_error_function.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_nca_nca_softmax_error_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/nca/nca_softmax_error_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTION_HPP
   #define MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <mlpack/core/math/make_alias.hpp>
   #include <mlpack/core/math/shuffle_data.hpp>
   
   namespace mlpack {
   namespace nca {
   
   template<typename MetricType = metric::SquaredEuclideanDistance>
   class SoftmaxErrorFunction
   {
    public:
     SoftmaxErrorFunction(const arma::mat& dataset,
                          const arma::Row<size_t>& labels,
                          MetricType metric = MetricType());
   
     void Shuffle();
   
     double Evaluate(const arma::mat& covariance);
   
     double Evaluate(const arma::mat& covariance,
                     const size_t begin,
                     const size_t batchSize = 1);
   
     void Gradient(const arma::mat& covariance, arma::mat& gradient);
   
     template <typename GradType>
     void Gradient(const arma::mat& covariance,
                   const size_t begin,
                   GradType& gradient,
                   const size_t batchSize = 1);
   
     const arma::mat GetInitialPoint() const;
   
     size_t NumFunctions() const { return dataset.n_cols; }
   
    private:
     arma::mat dataset;
     arma::Row<size_t> labels;
   
     MetricType metric;
   
     arma::mat lastCoordinates;
     arma::mat stretchedDataset;
     arma::vec p;
     arma::vec denominators;
   
     bool precalculated;
   
     void Precalculate(const arma::mat& coordinates);
   };
   
   } // namespace nca
   } // namespace mlpack
   
   // Include implementation.
   #include "nca_softmax_error_function_impl.hpp"
   
   #endif
