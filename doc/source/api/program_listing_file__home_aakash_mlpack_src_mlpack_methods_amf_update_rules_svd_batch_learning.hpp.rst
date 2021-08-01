
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_svd_batch_learning.hpp:

Program Listing for File svd_batch_learning.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_svd_batch_learning.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/update_rules/svd_batch_learning.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCH_LEARNING_HPP
   #define MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCH_LEARNING_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace amf {
   
   class SVDBatchLearning
   {
    public:
     SVDBatchLearning(double u = 0.0002,
                      double kw = 0,
                      double kh = 0,
                      double momentum = 0.9)
           : u(u), kw(kw), kh(kh), momentum(momentum)
     {
       // empty constructor
     }
   
     template<typename MatType>
     void Initialize(const MatType& dataset, const size_t rank)
     {
       const size_t n = dataset.n_rows;
       const size_t m = dataset.n_cols;
   
       mW.zeros(n, rank);
       mH.zeros(rank, m);
     }
   
     template<typename MatType>
     inline void WUpdate(const MatType& V,
                         arma::mat& W,
                         const arma::mat& H)
     {
       size_t n = V.n_rows;
       size_t m = V.n_cols;
   
       size_t r = W.n_cols;
   
       // initialize the momentum of this iteration.
       mW = momentum * mW;
   
       // Compute the step.
       arma::mat deltaW;
       deltaW.zeros(n, r);
       for (size_t i = 0; i < n; ++i)
       {
         for (size_t j = 0; j < m; ++j)
         {
           const double val = V(i, j);
           if (val != 0)
             deltaW.row(i) += (val - arma::dot(W.row(i), H.col(j))) *
                                               arma::trans(H.col(j));
         }
         // Add regularization.
         if (kw != 0)
           deltaW.row(i) -= kw * W.row(i);
       }
   
       // Add the step to the momentum.
       mW += u * deltaW;
       // Add the momentum to the W matrix.
       W += mW;
     }
   
     template<typename MatType>
     inline void HUpdate(const MatType& V,
                         const arma::mat& W,
                         arma::mat& H)
     {
       size_t n = V.n_rows;
       size_t m = V.n_cols;
   
       size_t r = W.n_cols;
   
       // Initialize the momentum of this iteration.
       mH = momentum * mH;
   
       // Compute the step.
       arma::mat deltaH;
       deltaH.zeros(r, m);
       for (size_t j = 0; j < m; ++j)
       {
         for (size_t i = 0; i < n; ++i)
         {
           const double val = V(i, j);
           if (val != 0)
             deltaH.col(j) += (val - arma::dot(W.row(i), H.col(j))) * W.row(i).t();
         }
         // Add regularization.
         if (kh != 0)
           deltaH.col(j) -= kh * H.col(j);
       }
   
       // Add this step to the momentum.
       mH += u * deltaH;
       // Add the momentum to H.
       H += mH;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(u));
       ar(CEREAL_NVP(kw));
       ar(CEREAL_NVP(kh));
       ar(CEREAL_NVP(momentum));
       ar(CEREAL_NVP(mW));
       ar(CEREAL_NVP(mH));
     }
   
    private:
     double u;
     double kw;
     double kh;
     double momentum;
   
     arma::mat mW;
     arma::mat mH;
   }; // class SVDBatchLearning
   
   
   template<>
   inline void SVDBatchLearning::WUpdate<arma::sp_mat>(const arma::sp_mat& V,
                                                       arma::mat& W,
                                                       const arma::mat& H)
   {
     const size_t n = V.n_rows;
     const size_t r = W.n_cols;
   
     mW = momentum * mW;
   
     arma::mat deltaW;
     deltaW.zeros(n, r);
   
     for (arma::sp_mat::const_iterator it = V.begin(); it != V.end(); ++it)
     {
       const size_t row = it.row();
       const size_t col = it.col();
       deltaW.row(it.row()) += (*it - arma::dot(W.row(row), H.col(col))) *
                                                arma::trans(H.col(col));
     }
   
     if (kw != 0)
       deltaW -= kw * W;
   
     mW += u * deltaW;
     W += mW;
   }
   
   template<>
   inline void SVDBatchLearning::HUpdate<arma::sp_mat>(const arma::sp_mat& V,
                                                       const arma::mat& W,
                                                       arma::mat& H)
   {
     const size_t m = V.n_cols;
     const size_t r = W.n_cols;
   
     mH = momentum * mH;
   
     arma::mat deltaH;
     deltaH.zeros(r, m);
   
     for (arma::sp_mat::const_iterator it = V.begin(); it != V.end(); ++it)
     {
       const size_t row = it.row();
       const size_t col = it.col();
       deltaH.col(col) += (*it - arma::dot(W.row(row), H.col(col))) *
           W.row(row).t();
     }
   
     if (kh != 0)
       deltaH -= kh * H;
   
     mH += u * deltaH;
     H += mH;
   }
   
   } // namespace amf
   } // namespace mlpack
   
   #endif // MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCH_LEARNING_HPP
