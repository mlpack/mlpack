
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_gmm_no_constraint.hpp:

Program Listing for File no_constraint.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_gmm_no_constraint.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/gmm/no_constraint.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_GMM_NO_CONSTRAINT_HPP
   #define MLPACK_METHODS_GMM_NO_CONSTRAINT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace gmm {
   
   class NoConstraint
   {
    public:
     static void ApplyConstraint(const arma::mat& /* covariance */) { }
   
     template<typename Archive>
     static void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace gmm
   } // namespace mlpack
   
   #endif
