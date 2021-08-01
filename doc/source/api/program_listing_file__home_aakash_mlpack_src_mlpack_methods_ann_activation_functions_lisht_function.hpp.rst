
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_lisht_function.hpp:

Program Listing for File lisht_function.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_lisht_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/lisht_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author = {Swalpa K. Roy, Suvojit Manna, Shiv R. Dubey and
              Bidyut B. Chaudhuri},
     title = {LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent
             Activation Function for Neural Networks},
     year = {2019}
   }
   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_LISHT_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_LISHT_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <algorithm>
   
   namespace mlpack {
   namespace ann  {
   
   class LiSHTFunction
   {
    public:
     static double Fn(const double x)
     {
       return x * std::tanh(x);
     }
   
     template <typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType &x, OutputVecType &y)
     {
       y = x % arma::tanh(x);
     }
   
     static double Deriv(const double y)
     {
       return std::tanh(y) + y * (1 - std::pow(std::tanh(y), 2));
     }
   
     template <typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType &y, OutputVecType &x)
     {
       x = arma::tanh(y) + y % (1 - arma::pow(arma::tanh(y), 2));
     }
   }; // class LishtFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
