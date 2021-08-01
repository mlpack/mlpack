
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_svd_incomplete_incremental_learning.hpp:

Program Listing for File svd_incomplete_incremental_learning.hpp
================================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_svd_incomplete_incremental_learning.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/update_rules/svd_incomplete_incremental_learning.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AMF_SVD_INCOMPLETE_INCREMENTAL_LEARNING_HPP
   #define MLPACK_METHODS_AMF_SVD_INCOMPLETE_INCREMENTAL_LEARNING_HPP
   
   namespace mlpack
   {
   namespace amf
   {
   
   class SVDIncompleteIncrementalLearning
   {
    public:
     SVDIncompleteIncrementalLearning(double u = 0.001,
                                      double kw = 0,
                                      double kh = 0)
             : u(u), kw(kw), kh(kh), currentUserIndex(0)
     {
       // Nothing to do.
     }
   
     template<typename MatType>
     void Initialize(const MatType& /* dataset */, const size_t /* rank */)
     {
       // Set the current user to 0.
       currentUserIndex = 0;
     }
   
     template<typename MatType>
     inline void WUpdate(const MatType& V,
                         arma::mat& W,
                         const arma::mat& H)
     {
       arma::mat deltaW;
       deltaW.zeros(V.n_rows, W.n_cols);
   
       // Iterate through all the rating by this user to update corresponding item
       // feature feature vector.
       for (size_t i = 0; i < V.n_rows; ++i)
       {
         const double val = V(i, currentUserIndex);
         // Update only if the rating is non-zero.
         if (val != 0)
         {
           deltaW.row(i) += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
               H.col(currentUserIndex).t();
         }
         // Add regularization.
         if (kw != 0)
           deltaW.row(i) -= kw * W.row(i);
       }
   
       W += u * deltaW;
     }
   
     template<typename MatType>
     inline void HUpdate(const MatType& V,
                         const arma::mat& W,
                         arma::mat& H)
     {
       arma::vec deltaH;
       deltaH.zeros(H.n_rows);
   
       // Iterate through all the rating by this user to update corresponding item
       // feature feature vector.
       for (size_t i = 0; i < V.n_rows; ++i)
       {
         const double val = V(i, currentUserIndex);
         // Update only if the rating is non-zero.
         if (val != 0)
         {
           deltaH += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
               W.row(i).t();
         }
       }
       // Add regularization.
       if (kh != 0)
         deltaH -= kh * H.col(currentUserIndex);
   
       // Update H matrix and move on to the next user.
       H.col(currentUserIndex++) += u * deltaH;
       currentUserIndex = currentUserIndex % V.n_cols;
     }
   
    private:
     double u;
     double kw;
     double kh;
   
     size_t currentUserIndex;
   };
   
   
   template<>
   inline void SVDIncompleteIncrementalLearning::WUpdate<arma::sp_mat>(
       const arma::sp_mat& V, arma::mat& W, const arma::mat& H)
   {
     arma::mat deltaW(V.n_rows, W.n_cols);
     deltaW.zeros();
     for (arma::sp_mat::const_iterator it = V.begin_col(currentUserIndex);
         it != V.end_col(currentUserIndex); ++it)
     {
       double val = *it;
       size_t i = it.row();
       deltaW.row(i) += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
           arma::trans(H.col(currentUserIndex));
       if (kw != 0) deltaW.row(i) -= kw * W.row(i);
     }
   
     W += u*deltaW;
   }
   
   template<>
   inline void SVDIncompleteIncrementalLearning::HUpdate<arma::sp_mat>(
       const arma::sp_mat& V, const arma::mat& W, arma::mat& H)
   {
     arma::mat deltaH(H.n_rows, 1);
     deltaH.zeros();
   
     for (arma::sp_mat::const_iterator it = V.begin_col(currentUserIndex);
         it != V.end_col(currentUserIndex); ++it)
     {
       double val = *it;
       size_t i = it.row();
       if ((val = V(i, currentUserIndex)) != 0)
       {
         deltaH += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
             arma::trans(W.row(i));
       }
     }
     if (kh != 0) deltaH -= kh * H.col(currentUserIndex);
   
     H.col(currentUserIndex++) += u * deltaH;
     currentUserIndex = currentUserIndex % V.n_cols;
   }
   
   } // namespace amf
   } // namespace mlpack
   
   #endif
