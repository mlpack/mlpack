
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_lars_lars_impl.hpp:

Program Listing for File lars_impl.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_lars_lars_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/lars/lars_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LARS_LARS_IMPL_HPP
   #define MLPACK_METHODS_LARS_LARS_IMPL_HPP
   
   #include "lars.hpp"
   
   namespace mlpack {
   namespace regression {
   
   template<typename Archive>
   void LARS::serialize(Archive& ar, const uint32_t /* version */)
   {
     // If we're loading, we have to use the internal storage.
     if (cereal::is_loading<Archive>())
     {
       matGram = &matGramInternal;
       ar(CEREAL_NVP(matGramInternal));
     }
     else
     {
       ar(cereal::make_nvp("matGramInternal",
           (const_cast<arma::mat&>(*matGram))));
     }
   
     ar(CEREAL_NVP(matUtriCholFactor));
     ar(CEREAL_NVP(useCholesky));
     ar(CEREAL_NVP(lasso));
     ar(CEREAL_NVP(lambda1));
     ar(CEREAL_NVP(elasticNet));
     ar(CEREAL_NVP(lambda2));
     ar(CEREAL_NVP(tolerance));
     ar(CEREAL_NVP(betaPath));
     ar(CEREAL_NVP(lambdaPath));
     ar(CEREAL_NVP(activeSet));
     ar(CEREAL_NVP(isActive));
     ar(CEREAL_NVP(ignoreSet));
     ar(CEREAL_NVP(isIgnored));
   }
   
   } // namespace regression
   } // namespace mlpack
   
   #endif
