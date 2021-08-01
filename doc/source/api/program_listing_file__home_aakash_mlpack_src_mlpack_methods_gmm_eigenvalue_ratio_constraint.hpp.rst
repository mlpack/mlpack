
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_gmm_eigenvalue_ratio_constraint.hpp:

Program Listing for File eigenvalue_ratio_constraint.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_gmm_eigenvalue_ratio_constraint.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/gmm/eigenvalue_ratio_constraint.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_GMM_EIGENVALUE_RATIO_CONSTRAINT_HPP
   #define MLPACK_METHODS_GMM_EIGENVALUE_RATIO_CONSTRAINT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace gmm {
   
   class EigenvalueRatioConstraint
   {
    public:
     EigenvalueRatioConstraint(const arma::vec& ratios) :
         // Make an alias of the ratios vector.  It will never be modified here.
         ratios(const_cast<double*>(ratios.memptr()), ratios.n_elem, false)
     {
       // Check validity of ratios.
       if (std::abs(ratios[0] - 1.0) > 1e-20)
         Log::Fatal << "EigenvalueRatioConstraint::EigenvalueRatioConstraint(): "
             << "first element of ratio vector is not 1.0!" << std::endl;
   
       for (size_t i = 1; i < ratios.n_elem; ++i)
       {
         if (ratios[i] > 1.0)
           Log::Fatal << "EigenvalueRatioConstraint::EigenvalueRatioConstraint(): "
               << "element " << i << " of ratio vector is greater than 1.0!"
               << std::endl;
         if (ratios[i] < 0.0)
           Log::Warn << "EigenvalueRatioConstraint::EigenvalueRatioConstraint(): "
               << "element " << i << " of ratio vectors is negative and will "
               << "probably cause the covariance to be non-invertible..."
               << std::endl;
       }
     }
   
     void ApplyConstraint(arma::mat& covariance) const
     {
       // Eigendecompose the matrix.
       arma::vec eigenvalues;
       arma::mat eigenvectors;
       covariance = arma::symmatu(covariance);
       if (!arma::eig_sym(eigenvalues, eigenvectors, covariance))
       {
         Log::Fatal << "applying to constraint could not be accomplished."
             << std::endl;
       }
   
       // Change the eigenvalues to what we are forcing them to be.  There
       // shouldn't be any negative eigenvalues anyway, so it doesn't matter if we
       // are suddenly forcing them to be positive.  If the first eigenvalue is
       // negative, well, there are going to be some problems later...
       eigenvalues = (eigenvalues[0] * ratios);
   
       // Reassemble the matrix.
       covariance = eigenvectors * arma::diagmat(eigenvalues) * eigenvectors.t();
     }
   
     void ApplyConstraint(arma::vec& diagCovariance) const
     {
       // The matrix is already eigendecomposed but we need to sort the elements.
       arma::uvec eigvalOrder = arma::sort_index(diagCovariance);
       arma::vec eigvals = diagCovariance(eigvalOrder);
   
       // Change the eigenvalues to what we are forcing them to be.  There
       // shouldn't be any negative eigenvalues anyway, so it doesn't matter if we
       // are suddenly forcing them to be positive.  If the first eigenvalue is
       // negative, well, there are going to be some problems later...
       eigvals = eigvals[0] * ratios;
   
       // Reassemble the matrix.
       for (size_t i = 0; i < eigvalOrder.n_elem; ++i)
         diagCovariance[eigvalOrder[i]] = eigvals[i];
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       // Strip the const for the sake of loading/saving.  This is the only time it
       // is modified (other than the constructor).
       ar(CEREAL_NVP(const_cast<arma::vec&>(ratios)));
     }
   
    private:
     const arma::vec ratios;
   };
   
   } // namespace gmm
   } // namespace mlpack
   
   #endif
