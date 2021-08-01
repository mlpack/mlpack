
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_he_init.hpp:

Program Listing for File he_init.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_he_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/he_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_INIT_RULES_HE_INIT_HPP
   #define MLPACK_METHODS_ANN_INIT_RULES_HE_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/random.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class HeInitialization
   {
    public:
     HeInitialization()
     {
       // Nothing to do here.
     }
   
     template <typename eT>
     void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
     {
       // He initialization rule says to initialize weights with random
       // values taken from a gaussian distribution with mean = 0 and
       // standard deviation = sqrt(2/rows), i.e. variance = (2/rows).
       const double variance = 2.0 / (double) rows;
   
       if (W.is_empty())
         W.set_size(rows, cols);
   
       // Multipling a random variable X with variance V(X) by some factor c,
       // then the variance V(cX) = (c^2) * V(X).
       W.imbue( [&]() { return sqrt(variance) * arma::randn(); } );
     }
   
     template <typename eT>
     void Initialize(arma::Mat<eT>& W)
     {
       // He initialization rule says to initialize weights with random
       // values taken from a gaussian distribution with mean = 0 and
       // standard deviation = sqrt(2 / rows), i.e. variance = (2 / rows).
       const double variance = 2.0 / (double) W.n_rows;
   
       if (W.is_empty())
         Log::Fatal << "Cannot initialize an empty matrix." << std::endl;
   
       // Multipling a random variable X with variance V(X) by some factor c,
       // then the variance V(cX) = (c^2) * V(X).
       W.imbue( [&]() { return sqrt(variance) * arma::randn(); } );
     }
   
     template <typename eT>
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
   
     template <typename eT>
     void Initialize(arma::Cube<eT> & W)
     {
       if (W.is_empty())
         Log::Fatal << "Cannot initialize an empty matrix" << std::endl;
   
       for (size_t i = 0; i < W.n_slices; ++i)
         Initialize(W.slice(i));
     }
   }; // class HeInitialization
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
