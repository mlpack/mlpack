
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_kathirvalavakumar_subavathi_init.hpp:

Program Listing for File kathirvalavakumar_subavathi_init.hpp
=============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_kathirvalavakumar_subavathi_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     title={A New Weight Initialization Method Using Cauchy’s Inequality Based
     on Sensitivity Analysis},
     author={T. Kathirvalavakumar and S. Subavathi},
     booktitle={Journal of Intelligent Learning Systems and Applications,
     Vol. 3 No. 4},
     year={2011}
   }
   
   #ifndef MLPACK_METHODS_ANN_INIT_RULES_KATHIRVALAVAKUMAR_SUBAVATHI_INIT_HPP
   #define MLPACK_METHODS_ANN_INIT_RULES_KATHIRVALAVAKUMAR_SUBAVATHI_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "init_rules_traits.hpp"
   #include "random_init.hpp"
   
   #include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
   
   #include <iostream>
   
   namespace mlpack {
   namespace ann  {
   
   class KathirvalavakumarSubavathiInitialization
   {
    public:
     template<typename eT>
     KathirvalavakumarSubavathiInitialization(const arma::Mat<eT>& data,
                                              const double s) : s(s)
     {
       dataSum = arma::sum(data % data);
     }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
     {
       arma::Row<eT> b = s * arma::sqrt(3 / (rows * dataSum));
       const double theta = b.min();
       RandomInitialization randomInit(-theta, theta);
       randomInit.Initialize(W, rows, cols);
     }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W)
     {
       arma::Row<eT> b = s * arma::sqrt(3 / (W.n_rows * dataSum));
       const double theta = b.min();
       RandomInitialization randomInit(-theta, theta);
       randomInit.Initialize(W);
     }
   
     template<typename eT>
     void Initialize(arma::Cube<eT>& W,
                     const size_t rows,
                     const size_t cols,
                     const size_t slices)
     {
       if (W.is_empty())
         W.set_size(rows, cols, slices);
   
       for (size_t i = 0; i < slices; ++i)
         Initialize(W.slice(i), rows, cols);
     }
   
     template<typename eT>
     void Initialize(arma::Cube<eT>& W)
     {
       if (W.is_empty())
         Log::Fatal << "Cannot initialize an empty cube." << std::endl;
   
       for (size_t i = 0; i < W.n_slices; ++i)
         Initialize(W.slice(i));
     }
   
    private:
     arma::rowvec dataSum;
   
     double s;
   }; // class KathirvalavakumarSubavathiInitialization
   
   template<>
   class InitTraits<KathirvalavakumarSubavathiInitialization>
   {
    public:
     static const bool UseLayer = false;
   };
   
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
