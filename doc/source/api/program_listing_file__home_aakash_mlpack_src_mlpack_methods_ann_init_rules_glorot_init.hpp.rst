
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_glorot_init.hpp:

Program Listing for File glorot_init.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_glorot_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/glorot_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_INIT_RULES_GLOROT_INIT_HPP
   #define MLPACK_METHODS_ANN_INIT_RULES_GLOROT_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "random_init.hpp"
   #include "gaussian_init.hpp"
   
   using namespace mlpack::math;
   
   namespace mlpack {
   namespace ann  {
   
   template<bool Uniform = true>
   class GlorotInitializationType
   {
    public:
     GlorotInitializationType()
     {
       // Nothing to do here.
     }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W,
                     const size_t rows,
                     const size_t cols);
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W);
   
     template<typename eT>
     void Initialize(arma::Cube<eT>& W,
                     const size_t rows,
                     const size_t cols,
                     const size_t slices);
   
     template<typename eT>
     void Initialize(arma::Cube<eT>& W);
   }; // class GlorotInitializationType
   
   template <>
   template<typename eT>
   inline void GlorotInitializationType<false>::Initialize(arma::Mat<eT>& W,
                                                          const size_t rows,
                                                          const size_t cols)
   {
     if (W.is_empty())
       W.set_size(rows, cols);
   
     double var = 2.0 / double(rows + cols);
     GaussianInitialization normalInit(0.0, var);
     normalInit.Initialize(W, rows, cols);
   }
   
   template <>
   template<typename eT>
   inline void GlorotInitializationType<false>::Initialize(arma::Mat<eT>& W)
   {
     if (W.is_empty())
       Log::Fatal << "Cannot initialize and empty matrix." << std::endl;
   
     double var = 2.0 / double(W.n_rows + W.n_cols);
     GaussianInitialization normalInit(0.0, var);
     normalInit.Initialize(W);
   }
   
   template <>
   template<typename eT>
   inline void GlorotInitializationType<true>::Initialize(arma::Mat<eT>& W,
                                                          const size_t rows,
                                                          const size_t cols)
   {
     if (W.is_empty())
       W.set_size(rows, cols);
   
     // Limit of distribution.
     double a = sqrt(6) / sqrt(rows + cols);
     RandomInitialization randomInit(-a, a);
     randomInit.Initialize(W, rows, cols);
   }
   
   template <>
   template<typename eT>
   inline void GlorotInitializationType<true>::Initialize(arma::Mat<eT>& W)
   {
     if (W.is_empty())
       Log::Fatal << "Cannot initialize an empty matrix." << std::endl;
   
     // Limit of distribution.
     double a = sqrt(6) / sqrt(W.n_rows + W.n_cols);
     RandomInitialization randomInit(-a, a);
     randomInit.Initialize(W);
   }
   
   template <bool Uniform>
   template<typename eT>
   inline void GlorotInitializationType<Uniform>::Initialize(arma::Cube<eT>& W,
                                                             const size_t rows,
                                                             const size_t cols,
                                                             const size_t slices)
   {
     if (W.is_empty())
       W.set_size(rows, cols, slices);
   
     for (size_t i = 0; i < slices; ++i)
       Initialize(W.slice(i), rows, cols);
   }
   
   template <bool Uniform>
   template<typename eT>
   inline void GlorotInitializationType<Uniform>::Initialize(arma::Cube<eT>& W)
   {
     if (W.is_empty())
       Log::Fatal << "Cannot initialize an empty matrix." << std::endl;
   
     for (size_t i = 0; i < W.n_slices; ++i)
       Initialize(W.slice(i));
   }
   
   // Convenience typedefs.
   
   using XavierInitialization = GlorotInitializationType<true>;
   
   using GlorotInitialization = GlorotInitializationType<false>;
   // Uses normal distribution
   } // namespace ann
   } // namespace mlpack
   
   #endif
