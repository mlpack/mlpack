
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_gmm_diagonal_constraint.hpp:

Program Listing for File diagonal_constraint.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_gmm_diagonal_constraint.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/gmm/diagonal_constraint.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_GMM_DIAGONAL_CONSTRAINT_HPP
   #define MLPACK_METHODS_GMM_DIAGONAL_CONSTRAINT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace gmm {
   
   class DiagonalConstraint
   {
    public:
     static void ApplyConstraint(arma::mat& covariance)
     {
       // Save the diagonal only.
       covariance = arma::diagmat(arma::clamp(covariance.diag(), 1e-10, DBL_MAX));
     }
   
     static void ApplyConstraint(arma::vec& diagCovariance)
     {
       // Although the covariance is already diagonal, clamp it to ensure each
       // value is at least 1e-10.
       diagCovariance = arma::clamp(diagCovariance, 1e-10, DBL_MAX);
     }
   
     template<typename Archive>
     static void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace gmm
   } // namespace mlpack
   
   #endif
