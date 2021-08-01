
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_poisson1_function.hpp:

Program Listing for File poisson1_function.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_poisson1_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/poisson1_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_POISSON1_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_POISSON1_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class Poisson1Function
   {
    public:
     static double Fn(const double x)
     {
       return (x - 1) * std::exp(-x);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = (x - 1) % arma::exp(-x);
     }
   
     static double Deriv(const double y)
     {
       return  std::exp(-y) + (1 - y) * std::exp(-y);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& x, OutputVecType& y)
     {
       y = arma::exp(-x) + (1 - x) % arma::exp(-x);
     }
   }; // class Poisson1Function
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
