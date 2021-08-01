
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_tanh_function.hpp:

Program Listing for File tanh_function.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_tanh_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/tanh_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class TanhFunction
   {
    public:
     static double Fn(const double x)
     {
       return std::tanh(x);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = arma::tanh(x);
     }
   
     static double Deriv(const double y)
     {
       return 1 - std::pow(y, 2);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& y, OutputVecType& x)
     {
       x = 1 - arma::pow(y, 2);
     }
   
     static double Inv(const double y)
     {
       return std::atanh(y);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Inv(const InputVecType& y, OutputVecType& x)
     {
       x = arma::atanh(y);
     }
   }; // class TanhFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
