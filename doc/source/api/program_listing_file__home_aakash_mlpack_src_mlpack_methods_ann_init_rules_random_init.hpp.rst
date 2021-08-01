
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_random_init.hpp:

Program Listing for File random_init.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_random_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/random_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_INIT_RULES_RANDOM_INIT_HPP
   #define MLPACK_METHODS_ANN_INIT_RULES_RANDOM_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class RandomInitialization
   {
    public:
     RandomInitialization(const double lowerBound = -1,
                          const double upperBound = 1) :
         lowerBound(lowerBound), upperBound(upperBound) { }
   
     RandomInitialization(const double bound) :
         lowerBound(-std::abs(bound)), upperBound(std::abs(bound)) { }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
     {
       if (W.is_empty())
         W.set_size(rows, cols);
   
       W.randu();
       W *= (upperBound - lowerBound);
       W += lowerBound;
     }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W)
     {
       if (W.is_empty())
         Log::Fatal << "Cannot initialize an empty matrix." << std::endl;
   
       W.randu();
       W *= (upperBound - lowerBound);
       W += lowerBound;
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
     double lowerBound;
   
     double upperBound;
   }; // class RandomInitialization
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
