
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_max_iteration_termination.hpp:

Program Listing for File max_iteration_termination.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_max_iteration_termination.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/termination_policies/max_iteration_termination.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AMF_TERMINATION_POLICIES_MAX_ITERATION_TERMINATION_HPP
   #define MLPACK_METHODS_AMF_TERMINATION_POLICIES_MAX_ITERATION_TERMINATION_HPP
   
   namespace mlpack {
   namespace amf {
   
   class MaxIterationTermination
   {
    public:
     MaxIterationTermination(const size_t maxIterations) :
         maxIterations(maxIterations),
         iteration(0)
     {
       if (maxIterations == 0)
         Log::Warn << "MaxIterationTermination::MaxIterationTermination(): given "
             << "number of iterations is 0, so algorithm will never terminate!"
             << std::endl;
     }
   
     template<typename MatType>
     void Initialize(const MatType& /* V */) { }
   
     bool IsConverged(const arma::mat& /* H */, const arma::mat& /* W */)
     {
       // Return true if we have performed the correct number of iterations.
       return (++iteration >= maxIterations);
     }
   
     size_t Index()
     {
       return (iteration > maxIterations) ? 0 : maxIterations - iteration;
     }
   
     size_t Iteration() const { return iteration; }
     size_t& Iteration() { return iteration; }
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
    private:
     size_t maxIterations;
     size_t iteration;
   };
   
   } // namespace amf
   } // namespace mlpack
   
   #endif
