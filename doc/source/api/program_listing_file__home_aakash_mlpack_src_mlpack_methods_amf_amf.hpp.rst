
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_amf.hpp:

Program Listing for File amf.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_amf.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/amf.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AMF_AMF_HPP
   #define MLPACK_METHODS_AMF_AMF_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include <mlpack/methods/amf/update_rules/nmf_mult_dist.hpp>
   #include <mlpack/methods/amf/update_rules/nmf_als.hpp>
   #include <mlpack/methods/amf/update_rules/svd_batch_learning.hpp>
   #include <mlpack/methods/amf/update_rules/svd_incomplete_incremental_learning.hpp>
   #include <mlpack/methods/amf/update_rules/svd_complete_incremental_learning.hpp>
   
   #include <mlpack/methods/amf/init_rules/random_init.hpp>
   #include <mlpack/methods/amf/init_rules/random_acol_init.hpp>
   
   #include <mlpack/methods/amf/termination_policies/simple_residue_termination.hpp>
   #include <mlpack/methods/amf/termination_policies/simple_tolerance_termination.hpp>
   
   namespace mlpack {
   namespace amf  {
   
   template<typename TerminationPolicyType = SimpleResidueTermination,
            typename InitializationRuleType = RandomAcolInitialization<>,
            typename UpdateRuleType = NMFMultiplicativeDistanceUpdate>
   class AMF
   {
    public:
     AMF(const TerminationPolicyType& terminationPolicy = TerminationPolicyType(),
         const InitializationRuleType& initializeRule = InitializationRuleType(),
         const UpdateRuleType& update = UpdateRuleType());
   
     template<typename MatType>
     double Apply(const MatType& V,
                  const size_t r,
                  arma::mat& W,
                  arma::mat& H);
   
     const TerminationPolicyType& TerminationPolicy() const
     { return terminationPolicy; }
     TerminationPolicyType& TerminationPolicy() { return terminationPolicy; }
   
     const InitializationRuleType& InitializeRule() const
     { return initializationRule; }
     InitializationRuleType& InitializeRule() { return initializationRule; }
   
     const UpdateRuleType& Update() const { return update; }
     UpdateRuleType& Update() { return update; }
   
    private:
     TerminationPolicyType terminationPolicy;
     InitializationRuleType initializationRule;
     UpdateRuleType update;
   }; // class AMF
   
   typedef amf::AMF<amf::SimpleResidueTermination,
                    amf::RandomAcolInitialization<>,
                    amf::NMFALSUpdate> NMFALSFactorizer;
   
   
   template<typename MatType = arma::mat>
   using SVDBatchFactorizer = amf::AMF<
       amf::SimpleResidueTermination,
       amf::RandomAcolInitialization<>,
       amf::SVDBatchLearning>;
   
   template<class MatType = arma::mat>
   using SVDIncompleteIncrementalFactorizer = amf::AMF<
       amf::SimpleResidueTermination,
       amf::RandomAcolInitialization<>,
       amf::SVDIncompleteIncrementalLearning>;
   template<class MatType = arma::mat>
   using SVDCompleteIncrementalFactorizer = amf::AMF<
       amf::SimpleResidueTermination,
       amf::RandomAcolInitialization<>,
       amf::SVDCompleteIncrementalLearning<MatType>>;
   } // namespace amf
   } // namespace mlpack
   
   // Include implementation.
   #include "amf_impl.hpp"
   
   #endif // MLPACK_METHODS_AMF_AMF_HPP
