
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_nmf_mult_dist.hpp:

Program Listing for File nmf_mult_dist.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_nmf_mult_dist.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/update_rules/nmf_mult_dist.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LMF_UPDATE_RULES_NMF_MULT_DIST_UPDATE_RULES_HPP
   #define MLPACK_METHODS_LMF_UPDATE_RULES_NMF_MULT_DIST_UPDATE_RULES_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace amf {
   
   class NMFMultiplicativeDistanceUpdate
   {
    public:
     // Empty constructor required for the UpdateRule template.
     NMFMultiplicativeDistanceUpdate() { }
   
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
       W = (W % (V * H.t())) / (W * H * H.t());
     }
   
     template<typename MatType>
     inline static void HUpdate(const MatType& V,
                                const arma::mat& W,
                                arma::mat& H)
     {
       H = (H % (W.t() * V)) / (W.t() * W * H);
     }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace amf
   } // namespace mlpack
   
   #endif
