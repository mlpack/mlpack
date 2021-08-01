
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_multi_quadratic_function.hpp:

Program Listing for File multi_quadratic_function.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_multi_quadratic_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/multi_quadratic_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_MULTIQUAD_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_MULTIQUAD_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class MultiQuadFunction
   {
    public:
     static double Fn(const double x)
     {
       return std::pow(1 + x * x, 0.5);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = arma::pow((1 + arma::pow(x, 2)), 0.5);
     }
   
     static double Deriv(const double y)
     {
       return  y / std::pow(1 + y * y, 0.5);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& x, OutputVecType& y)
     {
       y = x / arma::pow((1 + arma::pow(x, 2)), 0.5);
     }
   }; // class MultiquadFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
