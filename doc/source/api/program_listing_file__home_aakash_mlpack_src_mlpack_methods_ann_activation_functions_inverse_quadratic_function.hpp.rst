
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_inverse_quadratic_function.hpp:

Program Listing for File inverse_quadratic_function.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_inverse_quadratic_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/inverse_quadratic_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_INVERSE_QUAD_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_INVERSE_QUAD_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class InvQuadFunction
   {
    public:
     static double Fn(const double x)
     {
       return 1 / ( 1 + x * x);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = 1 / (1 + arma::pow(x, 2));
     }
   
     static double Deriv(const double y)
     {
       return  - 2 * y / std::pow(1 + std::pow(y, 2), 2);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& x, OutputVecType& y)
     {
       y = - 2 * x / arma::pow(1 + arma::pow(x, 2), 2);
     }
   }; // class InvQuadFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
