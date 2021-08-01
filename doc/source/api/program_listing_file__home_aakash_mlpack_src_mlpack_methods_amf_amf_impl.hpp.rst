
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_amf_impl.hpp:

Program Listing for File amf_impl.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_amf_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/amf_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   namespace mlpack {
   namespace amf {
   
   template<typename TerminationPolicyType,
            typename InitializationRuleType,
            typename UpdateRuleType>
   AMF<TerminationPolicyType, InitializationRuleType, UpdateRuleType>::AMF(
       const TerminationPolicyType& terminationPolicy,
       const InitializationRuleType& initializationRule,
       const UpdateRuleType& update) :
       terminationPolicy(terminationPolicy),
       initializationRule(initializationRule),
       update(update)
   { }
   
   template<typename TerminationPolicyType,
            typename InitializationRuleType,
            typename UpdateRuleType>
   template<typename MatType>
   double AMF<TerminationPolicyType, InitializationRuleType, UpdateRuleType>::
   Apply(const MatType& V,
         const size_t r,
         arma::mat& W,
         arma::mat& H)
   {
     // Initialize W and H.
     initializationRule.Initialize(V, r, W, H);
   
     Log::Info << "Initialized W and H." << std::endl;
   
     // initialize the update rule
     update.Initialize(V, r);
     // initialize the termination policy
     terminationPolicy.Initialize(V);
   
     // check if termination conditions are met
     while (!terminationPolicy.IsConverged(W, H))
     {
       // Update the values of W and H based on the update rules provided.
       update.WUpdate(V, W, H);
       update.HUpdate(V, W, H);
     }
   
     // get final residue and iteration count from termination policy
     const double residue = terminationPolicy.Index();
     const size_t iteration = terminationPolicy.Iteration();
   
     Log::Info << "AMF converged to residue of " << residue << " in "
         << iteration << " iterations." << std::endl;
   
     return residue;
   }
   
   } // namespace amf
   } // namespace mlpack
