
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_pca_decomposition_policies_exact_svd_method.hpp:

Program Listing for File exact_svd_method.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_pca_decomposition_policies_exact_svd_method.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_EXACT_SVD_METHOD_HPP
   #define MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_EXACT_SVD_METHOD_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace pca {
   
   class ExactSVDPolicy
   {
    public:
     void Apply(const arma::mat& data,
                const arma::mat& centeredData,
                arma::mat& transformedData,
                arma::vec& eigVal,
                arma::mat& eigvec,
                const size_t /* rank */)
     {
       // This matrix will store the right singular values; we do not need them.
       arma::mat v;
   
       // Do singular value decomposition.  Use the economical singular value
       // decomposition if the columns are much larger than the rows.
       if (data.n_rows < data.n_cols)
       {
         // Do economical singular value decomposition and compute only the left
         // singular vectors.
         arma::svd_econ(eigvec, eigVal, v, centeredData, 'l');
       }
       else
       {
         arma::svd(eigvec, eigVal, v, centeredData);
       }
   
       // Now we must square the singular values to get the eigenvalues.
       // In addition we must divide by the number of points, because the
       // covariance matrix is X * X' / (N - 1).
       eigVal %= eigVal / (data.n_cols - 1);
   
       // Project the samples to the principals.
       transformedData = arma::trans(eigvec) * centeredData;
     }
   };
   
   } // namespace pca
   } // namespace mlpack
   
   #endif
