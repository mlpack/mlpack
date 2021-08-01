
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_gelu_function.hpp:

Program Listing for File gelu_function.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_gelu_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/gelu_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GELU_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GELU_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class GELUFunction
   {
    public:
     static double Fn(const double x)
     {
       return 0.5 * x * (1 + std::tanh(std::sqrt(2 / M_PI) *
              (x + 0.044715 * std::pow(x, 3))));
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = 0.5 * x % (1 + arma::tanh(std::sqrt(2 / M_PI) *
           (x + 0.044715 * arma::pow(x, 3))));
     }
   
     static double Deriv(const double y)
     {
       return 0.5 * std::tanh(0.0356774 * std::pow(y, 3) + 0.797885 * y) +
              (0.0535161 * std::pow(y, 3) + 0.398942 * y) *
              std::pow(1 / std::cosh(0.0356774 * std::pow(y, 3) +
              0.797885 * y), 2) + 0.5;
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& y, OutputVecType& x)
     {
       x = 0.5 * arma::tanh(0.0356774 * arma::pow(y, 3) + 0.797885 * y) +
           (0.0535161 * arma::pow(y, 3) + 0.398942 * y) %
           arma::pow(1 / arma::cosh(0.0356774 * arma::pow(y, 3) +
           0.797885 * y), 2) + 0.5;
     }
   }; // class GELUFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
