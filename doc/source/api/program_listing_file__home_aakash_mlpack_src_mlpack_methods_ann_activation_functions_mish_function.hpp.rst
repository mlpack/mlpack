
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_mish_function.hpp:

Program Listing for File mish_function.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_mish_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/mish_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author = {Diganta Misra},
     title = {Mish: Self Regularized Non-Monotonic Neural Activation Function},
     year = {2019}
   }
   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_MISH_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_MISH_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <algorithm>
   
   namespace mlpack {
   namespace ann  {
   
   class MishFunction
   {
    public:
     static double Fn(const double x)
     {
       return x * (std::exp(2 * x) + 2 * std::exp(x)) /
              (2 + 2 * std::exp(x) + std::exp(2 * x));
     }
   
     template <typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType &x, OutputVecType &y)
     {
       y = x % (arma::exp(2 * x) + 2 * arma::exp(x)) /
           (2 + 2 * arma::exp(x) + arma::exp(2 * x));
     }
   
     static double Deriv(const double y)
     {
       return std::exp(y) * (4 * (y + 1) + std::exp(y) * (4 * y + 6) +
              4 * std::exp(2 * y) + std::exp(3 * y)) /
              std::pow(std::exp(2 * y) + 2 * std::exp(y) + 2, 2);
     }
   
     template <typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType &y, OutputVecType &x)
     {
       x = arma::exp(y) % (4 * (y + 1) + arma::exp(y) % (4 * y + 6) +
           4 * arma::exp(2 * y) + arma::exp(3 * y)) /
           arma::pow(arma::exp(2 * y) + 2 * arma::exp(y) + 2, 2);
     }
   }; // class MishFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
