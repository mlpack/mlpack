
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_block_krylov_svd_randomized_block_krylov_svd.hpp:

Program Listing for File randomized_block_krylov_svd.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_block_krylov_svd_randomized_block_krylov_svd.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/block_krylov_svd/randomized_block_krylov_svd.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_BLOCK_KRYLOV_SVD_RANDOMIZED_BLOCK_KRYLOV_SVD_HPP
   #define MLPACK_METHODS_BLOCK_KRYLOV_SVD_RANDOMIZED_BLOCK_KRYLOV_SVD_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace svd {
   
   class RandomizedBlockKrylovSVD
   {
    public:
     RandomizedBlockKrylovSVD(const arma::mat& data,
                              arma::mat& u,
                              arma::vec& s,
                              arma::mat& v,
                              const size_t maxIterations = 2,
                              const size_t rank = 0,
                              const size_t blockSize = 0);
   
     RandomizedBlockKrylovSVD(const size_t maxIterations = 2,
                              const size_t blockSize = 0);
   
     void Apply(const arma::mat& data,
                arma::mat& u,
                arma::vec& s,
                arma::mat& v,
                const size_t rank);
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     size_t BlockSize() const { return blockSize; }
     size_t& BlockSize() { return blockSize; }
   
    private:
     size_t maxIterations;
   
     size_t blockSize;
   };
   
   } // namespace svd
   } // namespace mlpack
   
   #endif
