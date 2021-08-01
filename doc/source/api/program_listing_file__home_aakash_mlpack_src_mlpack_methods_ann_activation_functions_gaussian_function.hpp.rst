
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_gaussian_function.hpp:

Program Listing for File gaussian_function.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_gaussian_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/gaussian_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GAUSSIAN_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GAUSSIAN_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class GaussianFunction
   {
    public:
     template<typename eT>
     static double Fn(const eT x)
     {
       return std::exp(-1 * std::pow(x, 2));
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = arma::exp(-1 * arma::pow(x, 2));
     }
   
     static double Deriv(const double y)
     {
       return 2 * -y * std::exp(-1 * std::pow(y, 2));
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& y, OutputVecType& x)
     {
       x = 2 * -y % arma::exp(-1 * arma::pow(y, 2));
     }
   }; // class GaussianFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
