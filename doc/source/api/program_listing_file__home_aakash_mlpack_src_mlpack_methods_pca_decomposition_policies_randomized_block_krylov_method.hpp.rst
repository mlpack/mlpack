
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_pca_decomposition_policies_randomized_block_krylov_method.hpp:

Program Listing for File randomized_block_krylov_method.hpp
===========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_pca_decomposition_policies_randomized_block_krylov_method.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/pca/decomposition_policies/randomized_block_krylov_method.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_RANDOMIZED_BLOCK_KRYLOV_HPP
   #define MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_RANDOMIZED_BLOCK_KRYLOV_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/block_krylov_svd/randomized_block_krylov_svd.hpp>
   
   namespace mlpack {
   namespace pca {
   
   class RandomizedBlockKrylovSVDPolicy
   {
    public:
     RandomizedBlockKrylovSVDPolicy(const size_t maxIterations = 2,
                                    const size_t blockSize = 0) :
         maxIterations(maxIterations),
         blockSize(blockSize)
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
   
       // Do singular value decomposition using the randomized block krylov SVD
       // algorithm.
       svd::RandomizedBlockKrylovSVD rsvd(maxIterations, blockSize);
       rsvd.Apply(centeredData, eigvec, eigVal, v, rank);
   
       // Now we must square the singular values to get the eigenvalues.
       // In addition we must divide by the number of points, because the
       // covariance matrix is X * X' / (N - 1).
       eigVal %= eigVal / (data.n_cols - 1);
   
       // Project the samples to the principals.
       transformedData = arma::trans(eigvec) * centeredData;
     }
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     size_t BlockSize() const { return blockSize; }
     size_t& BlockSize() { return blockSize; }
   
    private:
     size_t maxIterations;
   
     size_t blockSize;
   };
   
   } // namespace pca
   } // namespace mlpack
   
   #endif
