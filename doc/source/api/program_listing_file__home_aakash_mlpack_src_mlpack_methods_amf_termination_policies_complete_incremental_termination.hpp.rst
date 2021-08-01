
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_complete_incremental_termination.hpp:

Program Listing for File complete_incremental_termination.hpp
=============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_complete_incremental_termination.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/termination_policies/complete_incremental_termination.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AMF_COMPLETE_INCREMENTAL_TERMINATION_HPP
   #define MLPACK_METHODS_AMF_COMPLETE_INCREMENTAL_TERMINATION_HPP
   
   namespace mlpack {
   namespace amf {
   
   template<class TerminationPolicy>
   class CompleteIncrementalTermination
   {
    public:
     CompleteIncrementalTermination(
         TerminationPolicy tPolicy = TerminationPolicy()) :
         tPolicy(tPolicy), incrementalIndex(0), iteration(0)
     { /* Nothing to do here. */ }
   
     template<class MatType>
     void Initialize(const MatType& V)
     {
       tPolicy.Initialize(V);
   
       // Get the number of non-zero entries.
       incrementalIndex = arma::accu(V != 0);
       iteration = 0;
     }
   
     void Initialize(const arma::sp_mat& V)
     {
       tPolicy.Initialize(V);
   
       // Get number of non-zero entries
       incrementalIndex = V.n_nonzero;
       iteration = 0;
     }
   
     bool IsConverged(arma::mat& W, arma::mat& H)
     {
       // Increment iteration count.
       iteration++;
   
       // If iteration count is multiple of incremental index, return wrapped class
       // function.
       if (iteration % incrementalIndex == 0)
         return tPolicy.IsConverged(W, H);
       else
         return false;
     }
   
     const double& Index() const { return tPolicy.Index(); }
   
     const size_t& Iteration() const { return iteration; }
   
     const size_t& MaxIterations() const { return tPolicy.MaxIterations(); }
     size_t& MaxIterations() { return tPolicy.MaxIterations(); }
   
     const TerminationPolicy& TPolicy() const { return tPolicy; }
     TerminationPolicy& TPolicy() { return tPolicy; }
   
    private:
     TerminationPolicy tPolicy;
   
     size_t incrementalIndex;
     size_t iteration;
   }; // class CompleteIncrementalTermination
   
   } // namespace amf
   } // namespace mlpack
   
   #endif // MLPACK_METHODS_AMF_COMPLETE_INCREMENTAL_TERMINATION_HPP
