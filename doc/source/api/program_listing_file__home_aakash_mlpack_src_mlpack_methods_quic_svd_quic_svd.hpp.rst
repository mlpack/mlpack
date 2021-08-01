
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_quic_svd_quic_svd.hpp:

Program Listing for File quic_svd.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_quic_svd_quic_svd.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/quic_svd/quic_svd.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_QUIC_SVD_QUIC_SVD_HPP
   #define MLPACK_METHODS_QUIC_SVD_QUIC_SVD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/tree/cosine_tree/cosine_tree.hpp>
   
   namespace mlpack {
   namespace svd {
   
   class QUIC_SVD
   {
    public:
     QUIC_SVD(const arma::mat& dataset,
              arma::mat& u,
              arma::mat& v,
              arma::mat& sigma,
              const double epsilon = 0.03,
              const double delta = 0.1);
   
     void ExtractSVD(arma::mat& u, arma::mat& v, arma::mat& sigma);
   
    private:
     const arma::mat& dataset;
     arma::mat basis;
   };
   
   } // namespace svd
   } // namespace mlpack
   
   #endif
