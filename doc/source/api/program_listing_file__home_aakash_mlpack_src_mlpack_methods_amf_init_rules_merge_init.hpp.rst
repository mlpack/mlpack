
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_init_rules_merge_init.hpp:

Program Listing for File merge_init.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_init_rules_merge_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/init_rules/merge_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AMF_MERGE_INIT_HPP
   #define MLPACK_METHODS_AMF_MERGE_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace amf {
   
   template<typename WInitializationRuleType, typename HInitializationRuleType>
   class MergeInitialization
   {
    public:
     // Empty constructor required for the InitializeRule template
     MergeInitialization() { }
   
     // Initialize the MergeInitialization object with existing initialization
     // rules.
     MergeInitialization(const WInitializationRuleType& wInitRule,
                         const HInitializationRuleType& hInitRule) :
                         wInitializationRule(wInitRule),
                         hInitializationRule(hInitRule)
     { }
   
     template<typename MatType>
     inline void Initialize(const MatType& V,
                            const size_t r,
                            arma::mat& W,
                            arma::mat& H)
     {
       wInitializationRule.InitializeOne(V, r, W);
       hInitializationRule.InitializeOne(V, r, H, false);
     }
   
    private:
     // Initialization rule for W matrix
     WInitializationRuleType wInitializationRule;
     // Initialization rule for H matrix
     HInitializationRuleType hInitializationRule;
   };
   
   } // namespace amf
   } // namespace mlpack
   
   #endif
