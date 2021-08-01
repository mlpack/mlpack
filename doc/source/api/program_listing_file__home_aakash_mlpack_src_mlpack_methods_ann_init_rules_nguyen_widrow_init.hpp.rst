
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_nguyen_widrow_init.hpp:

Program Listing for File nguyen_widrow_init.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_nguyen_widrow_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     title={Improving the learning speed of 2-layer neural networks by choosing
     initial values of the adaptive weights},
     booktitle={Neural Networks, 1990., 1990 IJCNN International Joint
     Conference on},
     year={1990}
   }
   
   #ifndef MLPACK_METHODS_ANN_INIT_RULES_NGUYEN_WIDROW_INIT_HPP
   #define MLPACK_METHODS_ANN_INIT_RULES_NGUYEN_WIDROW_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "init_rules_traits.hpp"
   #include "random_init.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   class NguyenWidrowInitialization
   {
    public:
     NguyenWidrowInitialization(const double lowerBound = -0.5,
                                const double upperBound = 0.5) :
         lowerBound(lowerBound), upperBound(upperBound) { }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
     {
       RandomInitialization randomInit(lowerBound, upperBound);
       randomInit.Initialize(W, rows, cols);
   
       double beta = 0.7 * std::pow(cols, 1.0 / rows);
       W *= (beta / arma::norm(W));
     }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W)
     {
       RandomInitialization randomInit(lowerBound, upperBound);
       randomInit.Initialize(W);
   
       double beta = 0.7 * std::pow(W.n_cols, 1.0 / W.n_rows);
       W *= (beta / arma::norm(W));
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
         Log::Fatal << "Cannot initialize an empty matrix." << std::endl;
   
       for (size_t i = 0; i < W.n_slices; ++i)
         Initialize(W.slice(i));
     }
   
    private:
     double lowerBound;
   
     double upperBound;
   }; // class NguyenWidrowInitialization
   
   template<>
   class InitTraits<NguyenWidrowInitialization>
   {
    public:
     static const bool UseLayer = false;
   };
   
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
