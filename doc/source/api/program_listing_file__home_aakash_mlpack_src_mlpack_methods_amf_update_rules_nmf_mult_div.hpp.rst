
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_nmf_mult_div.hpp:

Program Listing for File nmf_mult_div.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_update_rules_nmf_mult_div.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/update_rules/nmf_mult_div.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LMF_UPDATE_RULES_NMF_MULT_DIV_HPP
   #define MLPACK_METHODS_LMF_UPDATE_RULES_NMF_MULT_DIV_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace amf {
   
   class NMFMultiplicativeDivergenceUpdate
   {
    public:
     // Empty constructor required for the WUpdateRule template.
     NMFMultiplicativeDivergenceUpdate() { }
   
     template<typename MatType>
     void Initialize(const MatType& /* dataset */, const size_t /* rank */)
     {
       // Nothing to do.
     }
   
     template<typename MatType>
     inline static void WUpdate(const MatType& V,
                                arma::mat& W,
                                const arma::mat& H)
     {
       // Simple implementation left in the header file.
       arma::mat t1;
       arma::rowvec t2;
   
       t1 = W * H;
       for (size_t i = 0; i < W.n_rows; ++i)
       {
         for (size_t j = 0; j < W.n_cols; ++j)
         {
           // Writing this as a single expression does not work as of Armadillo
           // 3.920.  This should be fixed in a future release, and then the code
           // below can be fixed.
           // t2 = H.row(j) % V.row(i) / t1.row(i);
           t2.set_size(H.n_cols);
           for (size_t k = 0; k < t2.n_elem; ++k)
           {
             t2(k) = H(j, k) * V(i, k) / t1(i, k);
           }
   
           W(i, j) = W(i, j) * sum(t2) / sum(H.row(j));
         }
       }
     }
   
     template<typename MatType>
     inline static void HUpdate(const MatType& V,
                               const arma::mat& W,
                               arma::mat& H)
     {
       // Simple implementation left in the header file.
       arma::mat t1;
       arma::colvec t2;
   
       t1 = W * H;
       for (size_t i = 0; i < H.n_rows; ++i)
       {
         for (size_t j = 0; j < H.n_cols; ++j)
         {
           // Writing this as a single expression does not work as of Armadillo
           // 3.920.  This should be fixed in a future release, and then the code
           // below can be fixed.
           // t2 = W.col(i) % V.col(j) / t1.col(j);
           t2.set_size(W.n_rows);
           for (size_t k = 0; k < t2.n_elem; ++k)
           {
             t2(k) = W(k, i) * V(k, j) / t1(k, j);
           }
   
           H(i, j) = H(i, j) * sum(t2) / sum(W.col(i));
         }
       }
     }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace amf
   } // namespace mlpack
   
   #endif
