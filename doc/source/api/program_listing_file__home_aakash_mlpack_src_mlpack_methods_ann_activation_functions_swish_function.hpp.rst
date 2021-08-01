
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_swish_function.hpp:

Program Listing for File swish_function.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_swish_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/swish_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SWISH_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SWISH_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class SwishFunction
   {
    public:
     static double Fn(const double x)
     {
       return x / (1.0 + std::exp(-x));
     }
   
     template<typename eT>
     static void Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
     {
       y = x / (1.0 + arma::exp(-x));
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y.set_size(arma::size(x));
   
       for (size_t i = 0; i < x.n_elem; ++i)
         y(i) = Fn(x(i));
     }
   
     static double Deriv(const double y)
     {
       return y / (1 + std::exp(-y)) + (1 - y / (1 + std::exp(-y))) /
                                                (1 + std::exp(-y));
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& y, OutputVecType& x)
     {
       x = y / (1 + arma::exp(-y)) + (1 - y / (1 + arma::exp(-y))) /
                                              (1 + arma::exp(-y));
     }
   }; // class SwishFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
