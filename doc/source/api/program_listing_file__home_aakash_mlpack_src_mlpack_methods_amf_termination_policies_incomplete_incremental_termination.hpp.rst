
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_incomplete_incremental_termination.hpp:

Program Listing for File incomplete_incremental_termination.hpp
===============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_incomplete_incremental_termination.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/termination_policies/incomplete_incremental_termination.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef _MLPACK_METHODS_AMF_INCOMPLETE_INCREMENTAL_TERMINATION_HPP
   #define _MLPACK_METHODS_AMF_INCOMPLETE_INCREMENTAL_TERMINATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace amf {
   
   template <class TerminationPolicy>
   class IncompleteIncrementalTermination
   {
    public:
     IncompleteIncrementalTermination(
         TerminationPolicy tPolicy = TerminationPolicy()) :
         tPolicy(tPolicy), incrementalIndex(0), iteration(0)
     { /* Nothing to do here. */ }
   
     template<class MatType>
     void Initialize(const MatType& V)
     {
       tPolicy.Initialize(V);
   
       // Initialize incremental index to number of rows.
       incrementalIndex = V.n_rows;
       iteration = 0;
     }
   
     bool IsConverged(arma::mat& W, arma::mat& H)
     {
       // increment iteration count
       iteration++;
   
       // If the iteration count is a multiple of incremental index, return the
       // wrapped termination policy result.
       if (iteration % incrementalIndex == 0)
         return tPolicy.IsConverged(W, H);
       else
         return false;
     }
   
     const double& Index() const { return tPolicy.Index(); }
   
     const size_t& Iteration() const { return iteration; }
   
     size_t MaxIterations() const { return tPolicy.MaxIterations(); }
     size_t& MaxIterations() { return tPolicy.MaxIterations(); }
   
     const TerminationPolicy& TPolicy() const { return tPolicy; }
     TerminationPolicy& TPolicy() { return tPolicy; }
   
    private:
     TerminationPolicy tPolicy;
   
     size_t incrementalIndex;
     size_t iteration;
   }; // class IncompleteIncrementalTermination
   
   } // namespace amf
   } // namespace mlpack
   
   #endif
