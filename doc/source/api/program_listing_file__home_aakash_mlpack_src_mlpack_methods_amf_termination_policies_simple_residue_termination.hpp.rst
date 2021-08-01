
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_simple_residue_termination.hpp:

Program Listing for File simple_residue_termination.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_simple_residue_termination.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/termination_policies/simple_residue_termination.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
   #define _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace amf {
   
   class SimpleResidueTermination
   {
    public:
     SimpleResidueTermination(const double minResidue = 1e-5,
                              const size_t maxIterations = 10000) :
       minResidue(minResidue),
       maxIterations(maxIterations),
       residue(0.0),
       iteration(0),
       normOld(0),
       nm(0)
     {
       // Nothing to do here.
     }
   
     template<typename MatType>
     void Initialize(const MatType& V)
     {
       // Initialize the things we keep track of.
       residue = DBL_MAX;
       iteration = 0;
       nm = V.n_rows * V.n_cols;
       // Remove history.
       normOld = 0;
     }
   
     bool IsConverged(arma::mat& W, arma::mat& H)
     {
       // Calculate the norm and compute the residue, but do it by hand, so as to
       // avoid calculating (W*H), which may be very large.
       double norm = 0.0;
       for (size_t j = 0; j < H.n_cols; ++j)
         norm += arma::norm(W * H.col(j), "fro");
       residue = fabs(normOld - norm) / normOld;
   
       // Store the norm.
       normOld = norm;
   
       // Increment iteration count
       iteration++;
       Log::Info << "Iteration " << iteration << "; residue " << residue << ".\n";
   
       // Check if termination criterion is met.
       // If maxIterations == 0, there is no iteration limit.
       return (residue < minResidue || iteration == maxIterations);
     }
   
     const double& Index() const { return residue; }
   
     const size_t& Iteration() const { return iteration; }
   
     const size_t& MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     const double& MinResidue() const { return minResidue; }
     double& MinResidue() { return minResidue; }
   
    public:
     double minResidue;
     size_t maxIterations;
   
     double residue;
     size_t iteration;
     double normOld;
   
     size_t nm;
   }; // class SimpleResidueTermination
   
   } // namespace amf
   } // namespace mlpack
   
   
   #endif // _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
