
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_rectifier_function.hpp:

Program Listing for File rectifier_function.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_rectifier_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/rectifier_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author = {Vinod Nair, Geoffrey E. Hinton},
     title = {Rectified Linear Units Improve Restricted Boltzmann Machines},
     year = {2010}
   }
   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_RECTIFIER_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_RECTIFIER_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <algorithm>
   
   namespace mlpack {
   namespace ann  {
   
   class RectifierFunction
   {
    public:
     static double Fn(const double x)
     {
       return std::max(0.0, x);
     }
   
     template<typename eT>
     static void Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
     {
       y.zeros(x.n_rows, x.n_cols);
       y = arma::max(y, x);
     }
   
     template<typename eT>
     static void Fn(const arma::Cube<eT>& x, arma::Cube<eT>& y)
     {
       y.zeros(x.n_rows, x.n_cols, x.n_slices);
       y = arma::max(y, x);
     }
   
     static double Deriv(const double x)
     {
       return (double)(x > 0);
     }
   
     template<typename InputType, typename OutputType>
     static void Deriv(const InputType& y, OutputType& x)
     {
       x.set_size(arma::size(y));
   
       for (size_t i = 0; i < y.n_elem; ++i)
         x(i) = Deriv(y(i));
     }
   }; // class RectifierFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
