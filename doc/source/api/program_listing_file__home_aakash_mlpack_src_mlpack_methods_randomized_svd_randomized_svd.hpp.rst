
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_randomized_svd_randomized_svd.hpp:

Program Listing for File randomized_svd.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_randomized_svd_randomized_svd.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/randomized_svd/randomized_svd.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RANDOMIZED_SVD_RANDOMIZED_SVD_HPP
   #define MLPACK_METHODS_RANDOMIZED_SVD_RANDOMIZED_SVD_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace svd {
   
   class RandomizedSVD
   {
    public:
     RandomizedSVD(const arma::mat& data,
                   arma::mat& u,
                   arma::vec& s,
                   arma::mat& v,
                   const size_t iteratedPower = 0,
                   const size_t maxIterations = 2,
                   const size_t rank = 0,
                   const double eps = 1e-7);
   
     RandomizedSVD(const size_t iteratedPower = 0,
                   const size_t maxIterations = 2,
                   const double eps = 1e-7);
   
     void Apply(const arma::sp_mat& data,
                arma::mat& u,
                arma::vec& s,
                arma::mat& v,
                const size_t rank);
   
     void Apply(const arma::mat& data,
                arma::mat& u,
                arma::vec& s,
                arma::mat& v,
                const size_t rank);
   
     template<typename MatType>
     void Apply(const MatType& data,
                arma::mat& u,
                arma::vec& s,
                arma::mat& v,
                const size_t rank,
                MatType rowMean)
     {
       if (iteratedPower == 0)
         iteratedPower = rank + 2;
   
       arma::mat R, Q, Qdata;
   
       // Apply the centered data matrix to a random matrix, obtaining Q.
       if (data.n_cols >= data.n_rows)
       {
         R = arma::randn<arma::mat>(data.n_rows, iteratedPower);
         Q = (data.t() * R) - arma::repmat(arma::trans(R.t() * rowMean),
             data.n_cols, 1);
       }
       else
       {
         R = arma::randn<arma::mat>(data.n_cols, iteratedPower);
         Q = (data * R) - (rowMean * (arma::ones(1, data.n_cols) * R));
       }
   
       // Form a matrix Q whose columns constitute a
       // well-conditioned basis for the columns of the earlier Q.
       if (maxIterations == 0)
       {
         arma::qr_econ(Q, v, Q);
       }
       else
       {
         arma::lu(Q, v, Q);
       }
   
       // Perform normalized power iterations.
       for (size_t i = 0; i < maxIterations; ++i)
       {
         if (data.n_cols >= data.n_rows)
         {
           Q = (data * Q) - rowMean * (arma::ones(1, data.n_cols) * Q);
           arma::lu(Q, v, Q);
           Q = (data.t() * Q) - arma::repmat(rowMean.t() * Q, data.n_cols, 1);
         }
         else
         {
           Q = (data.t() * Q) - arma::repmat(rowMean.t() * Q, data.n_cols, 1);
           arma::lu(Q, v, Q);
           Q = (data * Q) - (rowMean * (arma::ones(1, data.n_cols) * Q));
         }
   
         // Computing the LU decomposition is more efficient than computing the QR
         // decomposition, so we only use it in the last iteration, a pivoted QR
         // decomposition which renormalizes Q, ensuring that the columns of Q are
         // orthonormal.
         if (i < (maxIterations - 1))
         {
           arma::lu(Q, v, Q);
         }
         else
         {
           arma::qr_econ(Q, v, Q);
         }
       }
   
       // Do economical singular value decomposition and compute only the
       // approximations of the left singular vectors by using the centered data
       // applied to Q.
       if (data.n_cols >= data.n_rows)
       {
         Qdata = (data * Q) - rowMean * (arma::ones(1, data.n_cols) * Q);
         arma::svd_econ(u, s, v, Qdata);
         v = Q * v;
       }
       else
       {
         Qdata = (Q.t() * data) - arma::repmat(Q.t() * rowMean, 1,  data.n_cols);
         arma::svd_econ(u, s, v, Qdata);
         u = Q * u;
       }
     }
   
     size_t IteratedPower() const { return iteratedPower; }
     size_t& IteratedPower() { return iteratedPower; }
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     double Epsilon() const { return eps; }
     double& Epsilon() { return eps; }
   
    private:
     size_t iteratedPower;
   
     size_t maxIterations;
   
     double eps;
   };
   
   } // namespace svd
   } // namespace mlpack
   
   #endif
