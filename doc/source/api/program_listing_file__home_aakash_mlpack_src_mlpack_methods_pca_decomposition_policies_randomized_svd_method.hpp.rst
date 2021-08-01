
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_pca_decomposition_policies_randomized_svd_method.hpp:

Program Listing for File randomized_svd_method.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_pca_decomposition_policies_randomized_svd_method.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/pca/decomposition_policies/randomized_svd_method.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_RANDOMIZED_SVD_METHOD_HPP
   #define MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_RANDOMIZED_SVD_METHOD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/randomized_svd/randomized_svd.hpp>
   
   namespace mlpack {
   namespace pca {
   
   class RandomizedSVDPolicy
   {
    public:
     RandomizedSVDPolicy(const size_t iteratedPower = 0,
                         const size_t maxIterations = 2) :
         iteratedPower(iteratedPower),
         maxIterations(maxIterations)
     {
       /* Nothing to do here */
     }
   
     void Apply(const arma::mat& data,
                const arma::mat& centeredData,
                arma::mat& transformedData,
                arma::vec& eigVal,
                arma::mat& eigvec,
                const size_t rank)
     {
       // This matrix will store the right singular values; we do not need them.
       arma::mat v;
   
       // Do singular value decomposition using the randomized SVD algorithm.
       svd::RandomizedSVD rsvd(iteratedPower, maxIterations);
       rsvd.Apply(data, eigvec, eigVal, v, rank);
   
       // Now we must square the singular values to get the eigenvalues.
       // In addition we must divide by the number of points, because the
       // covariance matrix is X * X' / (N - 1).
       eigVal %= eigVal / (data.n_cols - 1);
   
       // Project the samples to the principals.
       transformedData = arma::trans(eigvec) * centeredData;
     }
   
     size_t IteratedPower() const { return iteratedPower; }
     size_t& IteratedPower() { return iteratedPower; }
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
    private:
     size_t iteratedPower;
   
     size_t maxIterations;
   };
   
   } // namespace pca
   } // namespace mlpack
   
   #endif
