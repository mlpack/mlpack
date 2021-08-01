
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_svd_complete_incremental_learning.hpp:

Program Listing for File svd_complete_incremental_learning.hpp
==============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_svd_complete_incremental_learning.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/update_rules/svd_complete_incremental_learning.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AMF_SVD_COMPLETE_INCREMENTAL_LEARNING_HPP
   #define MLPACK_METHODS_AMF_SVD_COMPLETE_INCREMENTAL_LEARNING_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack
   {
   namespace amf
   {
   
   template <class MatType>
   class SVDCompleteIncrementalLearning
   {
    public:
     SVDCompleteIncrementalLearning(double u = 0.0001,
                                    double kw = 0,
                                    double kh = 0)
               : u(u), kw(kw), kh(kh), currentUserIndex(0), currentItemIndex(0)
     {
       // Nothing to do.
     }
   
     void Initialize(const MatType& /* dataset */, const size_t /* rank */)
     {
       // Initialize the current score counters.
       currentUserIndex = 0;
       currentItemIndex = 0;
     }
   
     inline void WUpdate(const MatType& V,
                         arma::mat& W,
                         const arma::mat& H)
     {
       arma::mat deltaW;
       deltaW.zeros(1, W.n_cols);
   
       // Loop until a non-zero entry is found.
       while (true)
       {
         const double val = V(currentItemIndex, currentUserIndex);
         // Update feature vector if current entry is non-zero and break the loop.
         if (val != 0)
         {
           deltaW += (val - arma::dot(W.row(currentItemIndex),
               H.col(currentUserIndex))) * H.col(currentUserIndex).t();
   
           // Add regularization.
           if (kw != 0)
             deltaW -= kw * W.row(currentItemIndex);
           break;
         }
       }
   
       W.row(currentItemIndex) += u * deltaW;
     }
   
     inline void HUpdate(const MatType& V,
                         const arma::mat& W,
                         arma::mat& H)
     {
       arma::mat deltaH;
       deltaH.zeros(H.n_rows, 1);
   
       const double val = V(currentItemIndex, currentUserIndex);
   
       // Update H matrix based on the non-zero entry found in WUpdate function.
       deltaH += (val - arma::dot(W.row(currentItemIndex),
           H.col(currentUserIndex))) * W.row(currentItemIndex).t();
       // Add regularization.
       if (kh != 0)
         deltaH -= kh * H.col(currentUserIndex);
   
       // Move on to the next entry.
       currentUserIndex = currentUserIndex + 1;
       if (currentUserIndex == V.n_rows)
       {
         currentUserIndex = 0;
         currentItemIndex = (currentItemIndex + 1) % V.n_cols;
       }
   
       H.col(currentUserIndex++) += u * deltaH;
     }
   
    private:
     double u;
     double kw;
     double kh;
   
     size_t currentUserIndex;
     size_t currentItemIndex;
   };
   
   
   template<>
   class SVDCompleteIncrementalLearning<arma::sp_mat>
   {
    public:
     SVDCompleteIncrementalLearning(double u = 0.01,
                                    double kw = 0,
                                    double kh = 0)
               : u(u), kw(kw), kh(kh), n(0), m(0), it(NULL), isStart(false)
       {}
   
     ~SVDCompleteIncrementalLearning()
     {
       delete it;
     }
   
     void Initialize(const arma::sp_mat& dataset, const size_t rank)
     {
       (void)rank;
       n = dataset.n_rows;
       m = dataset.n_cols;
   
       it = new arma::sp_mat::const_iterator(dataset.begin());
       isStart = true;
     }
   
     inline void WUpdate(const arma::sp_mat& V,
                         arma::mat& W,
                         const arma::mat& H)
     {
       if (!isStart)
           ++(*it);
       else isStart = false;
   
       if (*it == V.end())
       {
           delete it;
           it = new arma::sp_mat::const_iterator(V.begin());
       }
   
       size_t currentUserIndex = it->col();
       size_t currentItemIndex = it->row();
   
       arma::mat deltaW(1, W.n_cols);
       deltaW.zeros();
   
       deltaW += (**it - arma::dot(W.row(currentItemIndex),
           H.col(currentUserIndex))) * arma::trans(H.col(currentUserIndex));
       if (kw != 0) deltaW -= kw * W.row(currentItemIndex);
   
       W.row(currentItemIndex) += u*deltaW;
     }
   
     inline void HUpdate(const arma::sp_mat& /* V */,
                         const arma::mat& W,
                         arma::mat& H)
     {
       arma::mat deltaH(H.n_rows, 1);
       deltaH.zeros();
   
       size_t currentUserIndex = it->col();
       size_t currentItemIndex = it->row();
   
       deltaH += (**it - arma::dot(W.row(currentItemIndex),
           H.col(currentUserIndex))) * arma::trans(W.row(currentItemIndex));
       if (kh != 0) deltaH -= kh * H.col(currentUserIndex);
   
       H.col(currentUserIndex) += u * deltaH;
     }
   
    private:
     double u;
     double kw;
     double kh;
   
     size_t n;
     size_t m;
   
     arma::sp_mat dummy;
     arma::sp_mat::const_iterator* it;
   
     bool isStart;
   }; // class SVDCompleteIncrementalLearning
   
   } // namespace amf
   } // namespace mlpack
   
   #endif
