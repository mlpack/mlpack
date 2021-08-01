
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_gmm_positive_definite_constraint.hpp:

Program Listing for File positive_definite_constraint.hpp
=========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_gmm_positive_definite_constraint.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/gmm/positive_definite_constraint.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_GMM_POSITIVE_DEFINITE_CONSTRAINT_HPP
   #define MLPACK_METHODS_GMM_POSITIVE_DEFINITE_CONSTRAINT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace gmm {
   
   class PositiveDefiniteConstraint
   {
    public:
     static void ApplyConstraint(arma::mat& covariance)
     {
       // What we want to do is make sure that the matrix is positive definite and
       // that the condition number isn't too large.  We also need to ensure that
       // the covariance matrix is not too close to zero (hence, we ensure that all
       // eigenvalues are at least 1e-50).
       arma::vec eigval;
       arma::mat eigvec;
       covariance = arma::symmatu(covariance);
       if (!arma::eig_sym(eigval, eigvec, covariance))
       {
         Log::Fatal << "applying to constraint could not be accomplished."
             << std::endl;
       }
   
       // If the matrix is not positive definite or if the condition number is
       // large, we must project it back onto the cone of positive definite
       // matrices with reasonable condition number (I'm picking 1e5 here, not for
       // any particular reason).
       if ((eigval[0] < 0.0) || ((eigval[eigval.n_elem - 1] / eigval[0]) > 1e5) ||
           (eigval[eigval.n_elem - 1] < 1e-50))
       {
         // Project any negative eigenvalues back to non-negative, and project any
         // too-small eigenvalues to a large enough value.  Make them as small as
         // possible to satisfy our constraint on the condition number.
         const double minEigval = std::max(eigval[eigval.n_elem - 1] / 1e5, 1e-50);
         for (size_t i = 0; i < eigval.n_elem; ++i)
           eigval[i] = std::max(eigval[i], minEigval);
   
         // Now reassemble the covariance matrix.
         covariance = eigvec * arma::diagmat(eigval) * eigvec.t();
       }
     }
   
     static void ApplyConstraint(arma::vec& diagCovariance)
     {
       // If the matrix is not positive definite or if the condition number is
       // large, we must project it back onto the cone of positive definite
       // matrices with reasonable condition number (I'm picking 1e5 here, not for
       // any particular reason).
       double maxEigval = -DBL_MAX;
       for (size_t i = 0; i < diagCovariance.n_elem; ++i)
       {
         if (diagCovariance[i] > maxEigval)
           maxEigval = diagCovariance[i];
       }
   
       for (size_t i = 0; i < diagCovariance.n_elem; ++i)
       {
         if ((diagCovariance[i] < 0.0) || ((maxEigval / diagCovariance[i]) > 1e5)
             || (maxEigval < 1e-50))
         {
           diagCovariance[i] = std::max(maxEigval / 1e5, 1e-50);
         }
       }
     }
   
     template<typename Archive>
     static void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace gmm
   } // namespace mlpack
   
   #endif
   
