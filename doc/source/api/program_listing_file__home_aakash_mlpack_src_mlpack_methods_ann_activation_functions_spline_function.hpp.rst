
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_spline_function.hpp:

Program Listing for File spline_function.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_spline_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/spline_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SPLINE_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SPLINE_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class SplineFunction
   {
    public:
     static double Fn(const double x)
     {
       return std::pow(x, 2) * std::log(1 + x);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = arma::pow(x, 2) % arma::log(1 + x);
     }
   
     static double Deriv(const double y)
     {
       return  2 * y * std::log(1 + y) + std::pow(y, 2) / (1 + y);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& x, OutputVecType& y)
     {
       y = 2 * x % arma::log(1 + x) + arma::pow(x, 2) / (1 + x);
     }
   }; // class SplineFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
