
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_gaussian_init.hpp:

Program Listing for File gaussian_init.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_gaussian_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/gaussian_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_INIT_RULES_GAUSSIAN_INIT_HPP
   #define MLPACK_METHODS_ANN_INIT_RULES_GAUSSIAN_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/random.hpp>
   
   using namespace mlpack::math;
   
   namespace mlpack {
   namespace ann  {
   
   class GaussianInitialization
   {
    public:
     GaussianInitialization(const double mean = 0, const double variance = 1) :
         mean(mean), variance(variance)
     {
       // Nothing to do here.
     }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W,
                     const size_t rows,
                     const size_t cols)
     {
       if (W.is_empty())
         W.set_size(rows, cols);
   
       W.imbue( [&]() { return arma::as_scalar(RandNormal(mean, variance)); } );
     }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W)
     {
       if (W.is_empty())
         Log::Fatal << "Cannot initialize an empty matrix." << std::endl;
   
       W.imbue( [&]() { return arma::as_scalar(RandNormal(mean, variance)); } );
     }
   
     template<typename eT>
     void Initialize(arma::Cube<eT> & W,
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
     void Initialize(arma::Cube<eT> & W)
     {
       if (W.is_empty())
         Log::Fatal << "Cannot initialize an empty matrix." << std::endl;
   
       for (size_t i = 0; i < W.n_slices; ++i)
         Initialize(W.slice(i));
     }
   
    private:
     double mean;
   
     double variance;
   }; // class GaussianInitialization
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
