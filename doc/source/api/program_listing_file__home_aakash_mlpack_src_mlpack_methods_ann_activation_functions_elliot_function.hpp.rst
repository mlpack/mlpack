
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_elliot_function.hpp:

Program Listing for File elliot_function.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_elliot_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/elliot_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     title = {A better activation function for artificial neural networks},
     author = {Elliott, David L},
     url = {https://drum.lib.umd.edu/bitstream/handle/1903/5355/TR_93-8.pdf}
     year = {1993}
   }
   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ELLIOT_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ELLIOT_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class ElliotFunction
   {
    public:
     static double Fn(const double x)
     {
       return x / (1.0 + std::abs(x));
     }
   
     template <typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType &x, OutputVecType &y)
     {
       y = x / (1.0 + arma::abs(x));
     }
   
     static double Deriv(const double y)
     {
       return 1.0 / std::pow(1.0 + std::abs(y), 2);
     }
   
     template <typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType &y, OutputVecType &x)
     {
       x = 1.0 / arma::pow(1.0 + arma::abs(y), 2);
     }
   }; // class ElliotFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
