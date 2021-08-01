
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_pca_decomposition_policies_quic_svd_method.hpp:

Program Listing for File quic_svd_method.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_pca_decomposition_policies_quic_svd_method.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/pca/decomposition_policies/quic_svd_method.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_QUIC_SVD_METHOD_HPP
   #define MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_QUIC_SVD_METHOD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/quic_svd/quic_svd.hpp>
   
   namespace mlpack {
   namespace pca {
   
   class QUICSVDPolicy
   {
    public:
     QUICSVDPolicy(const double epsilon = 0.03, const double delta = 0.1) :
          epsilon(epsilon),
          delta(delta)
     {
       /* Nothing to do here */
     }
   
     void Apply(const arma::mat& data,
                const arma::mat& centeredData,
                arma::mat& transformedData,
                arma::vec& eigVal,
                arma::mat& eigvec,
                const size_t /* rank */)
     {
       // This matrix will store the right singular values; we do not need them.
       arma::mat v, sigma;
   
       // Do singular value decomposition using the QUIC-SVD algorithm.
       svd::QUIC_SVD quicsvd(centeredData, eigvec, v, sigma, epsilon, delta);
   
       // Now we must square the singular values to get the eigenvalues.
       // In addition we must divide by the number of points, because the
       // covariance matrix is X * X' / (N - 1).
       eigVal = arma::pow(arma::diagvec(sigma), 2) / (data.n_cols - 1);
   
       // Project the samples to the principals.
       transformedData = arma::trans(eigvec) * centeredData;
     }
   
     double Epsilon() const { return epsilon; }
     double& Epsilon() { return epsilon; }
   
     double Delta() const { return delta; }
     double& Delta() { return delta; }
   
    private:
     double epsilon;
   
     double delta;
   };
   
   } // namespace pca
   } // namespace mlpack
   
   #endif
