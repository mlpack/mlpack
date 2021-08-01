
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_svd_wrapper_impl.hpp:

Program Listing for File svd_wrapper_impl.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_svd_wrapper_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/svd_wrapper_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SVDWRAPPER_IMPL_HPP
   #define MLPACK_METHODS_SVDWRAPPER_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "svd_wrapper.hpp"
   
   namespace mlpack {
   namespace cf {
   
   template<class Factorizer>
   double SVDWrapper<Factorizer>::Apply(const arma::mat& V,
                                        arma::mat& W,
                                        arma::mat& sigma,
                                        arma::mat& H) const
   {
     // get svd factorization
     arma::vec E;
     factorizer.Apply(W, E, H, V);
   
     // construct sigma matrix
     sigma.zeros(V.n_rows, V.n_cols);
   
     for (size_t i = 0; i < sigma.n_rows && i < sigma.n_cols; ++i)
       sigma(i, i) = E(i, 0);
   
     arma::mat V_rec = W * sigma * arma::trans(H);
   
     // return normalized frobenius error
     return arma::norm(V - V_rec, "fro") / arma::norm(V, "fro");
   }
   
   template<>
   double SVDWrapper<DummyClass>::Apply(const arma::mat& V,
                                        arma::mat& W,
                                        arma::mat& sigma,
                                        arma::mat& H) const
   {
     // get svd factorization
     arma::vec E;
     arma::svd(W, E, H, V);
   
     // construct sigma matrix
     sigma.zeros(V.n_rows, V.n_cols);
   
     for (size_t i = 0; i < sigma.n_rows && i < sigma.n_cols; ++i)
       sigma(i, i) = E(i, 0);
   
     arma::mat V_rec = W * sigma * arma::trans(H);
   
     // return normalized frobenius error
     return arma::norm(V - V_rec, "fro") / arma::norm(V, "fro");
   }
   
   template<class Factorizer>
   double SVDWrapper<Factorizer>::Apply(const arma::mat& V,
                            size_t r,
                            arma::mat& W,
                            arma::mat& H) const
   {
     // check if the given rank is valid
     if (r > V.n_rows || r > V.n_cols)
     {
       Log::Info << "Rank " << r << ", given for decomposition is invalid."
           << std::endl;
   
       r = (V.n_rows > V.n_cols) ? V.n_cols : V.n_rows;
       Log::Info << "Setting decomposition rank to " << r << std::endl;
     }
   
     // get svd factorization
     arma::vec sigma;
     factorizer.Apply(W, sigma, H, V);
   
     // remove the part of W and H depending upon the value of rank
     W = W.submat(0, 0, W.n_rows - 1, r - 1);
     H = H.submat(0, 0, H.n_cols - 1, r - 1);
   
     // take only required eigenvalues
     sigma = sigma.subvec(0, r - 1);
   
     // eigenvalue matrix is multiplied to W
     // it can either be multiplied to H matrix
     W = W * arma::diagmat(sigma);
   
     // take transpose of the matrix H as required by CF module
     H = arma::trans(H);
   
     // reconstruct the matrix
     arma::mat V_rec = W * H;
   
     // return the normalized frobenius norm
     return arma::norm(V - V_rec, "fro") / arma::norm(V, "fro");
   }
   
   template<>
   double SVDWrapper<DummyClass>::Apply(const arma::mat& V,
                                        size_t r,
                                        arma::mat& W,
                                        arma::mat& H) const
   {
     // check if the given rank is valid
     if (r > V.n_rows || r > V.n_cols)
     {
       Log::Info << "Rank " << r << ", given for decomposition is invalid."
           << std::endl;
   
       r = (V.n_rows > V.n_cols) ? V.n_cols : V.n_rows;
       Log::Info << "Setting decomposition rank to " << r << std::endl;
     }
   
     // get svd factorization
     arma::vec sigma;
     arma::svd(W, sigma, H, V);
   
     // remove the part of W and H depending upon the value of rank
     W = W.submat(0, 0, W.n_rows - 1, r - 1);
     H = H.submat(0, 0, H.n_cols - 1, r - 1);
   
     // take only required eigenvalues
     sigma = sigma.subvec(0, r - 1);
   
     // eigenvalue matrix is multiplied to W
     // it can either be multiplied to H matrix
     W = W * arma::diagmat(sigma);
   
     // take transpose of the matrix H as required by CF module
     H = arma::trans(H);
   
     // reconstruct the matrix
     arma::mat V_rec = W * H;
   
     // return the normalized frobenius norm
     return arma::norm(V - V_rec, "fro") / arma::norm(V, "fro");
   }
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
