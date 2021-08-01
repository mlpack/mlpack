
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_regularized_svd_regularized_svd_function.hpp:

Program Listing for File regularized_svd_function.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_regularized_svd_regularized_svd_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/regularized_svd/regularized_svd_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_FUNCTION_SVD_HPP
   #define MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_FUNCTION_SVD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <ensmallen.hpp>
   
   namespace mlpack {
   namespace svd {
   
   template <typename MatType = arma::mat>
   class RegularizedSVDFunction
   {
    public:
     RegularizedSVDFunction(const MatType& data,
                            const size_t rank,
                            const double lambda);
   
     void Shuffle();
   
     double Evaluate(const arma::mat& parameters) const;
   
     double Evaluate(const arma::mat& parameters,
                     const size_t start,
                     const size_t batchSize = 1) const;
   
     void Gradient(const arma::mat& parameters,
                   arma::mat& gradient) const;
   
     template <typename GradType>
     void Gradient(const arma::mat& parameters,
                   const size_t start,
                   GradType& gradient,
                   const size_t batchSize = 1) const;
   
     const arma::mat& GetInitialPoint() const { return initialPoint; }
   
     const arma::mat& Dataset() const { return data; }
   
     size_t NumFunctions() const { return data.n_cols; }
   
     size_t NumUsers() const { return numUsers; }
   
     size_t NumItems() const { return numItems; }
   
     double Lambda() const { return lambda; }
   
     size_t Rank() const { return rank; }
   
    private:
     MatType data;
     arma::mat initialPoint;
     size_t rank;
     double lambda;
     size_t numUsers;
     size_t numItems;
   };
   
   } // namespace svd
   } // namespace mlpack
   
   namespace ens {
   
     template <>
     template <>
     inline double StandardSGD::Optimize(
         mlpack::svd::RegularizedSVDFunction<arma::mat>& function,
         arma::mat& parameters);
   
     template <>
     template <>
     inline double ParallelSGD<ExponentialBackoff>::Optimize(
         mlpack::svd::RegularizedSVDFunction<arma::mat>& function,
         arma::mat& parameters);
   
   } // namespace ens
   
   #include "regularized_svd_function_impl.hpp"
   
   #endif
