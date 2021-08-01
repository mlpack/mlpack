
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_quadratic_function.hpp:

Program Listing for File quadratic_function.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_quadratic_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/quadratic_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_QUAD_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_QUAD_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class QuadraticFunction
   {
    public:
     static double Fn(const double x)
     {
       return std::pow(x, 2);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = arma::pow(x, 2);
     }
   
     static double Deriv(const double y)
     {
       return 2 * y;
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& x, OutputVecType& y)
     {
       y = 2 * x;
     }
   }; // class QUADFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
