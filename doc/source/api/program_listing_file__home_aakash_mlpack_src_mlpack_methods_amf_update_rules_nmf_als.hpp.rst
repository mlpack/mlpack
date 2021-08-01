
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_nmf_als.hpp:

Program Listing for File nmf_als.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_nmf_als.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/update_rules/nmf_als.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LMF_UPDATE_RULES_NMF_ALS_HPP
   #define MLPACK_METHODS_LMF_UPDATE_RULES_NMF_ALS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace amf {
   
   class NMFALSUpdate
   {
    public:
     NMFALSUpdate() { }
   
     template<typename MatType>
     void Initialize(const MatType& /* dataset */, const size_t /* rank */)
     {
       // Nothing to do.
     }
   
     template<typename MatType>
     inline static void WUpdate(const MatType& V,
                                arma::mat& W,
                                const arma::mat& H)
     {
       // The call to inv() sometimes fails; so we are using the psuedoinverse.
       // W = (inv(H * H.t()) * H * V.t()).t();
       W = V * H.t() * pinv(H * H.t());
   
       // Set all negative numbers to machine epsilon.
       for (size_t i = 0; i < W.n_elem; ++i)
       {
         if (W(i) < 0.0)
         {
           W(i) = 0.0;
         }
       }
     }
   
     template<typename MatType>
     inline static void HUpdate(const MatType& V,
                                const arma::mat& W,
                                arma::mat& H)
     {
       H = pinv(W.t() * W) * W.t() * V;
   
       // Set all negative numbers to 0.
       for (size_t i = 0; i < H.n_elem; ++i)
       {
         if (H(i) < 0.0)
         {
           H(i) = 0.0;
         }
       }
     }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   }; // class NMFALSUpdate
   
   } // namespace amf
   } // namespace mlpack
   
   #endif
