
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_tanh_exponential_function.hpp:

Program Listing for File tanh_exponential_function.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_tanh_exponential_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/tanh_exponential_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

      title = {TanhExp: A Smooth Activation Function with High Convergence Speed
               for Lightweight Neural Networks},
      author = {Xinyu Liu and Xiaoguang Di},
      year = {2020},
      url = {https://arxiv.org/pdf/2003.09855v2.pdf},
      eprint = {2003.09855v2},
      archivePrefix = {arXiv},
      primaryClass = {cs.LG} }
   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_EXPONENTIAL_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_TANH_EXPONENTIAL_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class TanhExpFunction
   {
    public:
     static double Fn(const double x)
     {
       return x * std::tanh(std::exp(x));
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = x % arma::tanh(arma::exp(x));
     }
   
     static double Deriv(const double y)
     {
       return std::tanh(std::exp(y)) - y * std::exp(y) *
           (std::pow(std::tanh(std::exp(y)), 2) - 1);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& y, OutputVecType& x)
     {
       x = arma::tanh(arma::exp(y)) - y % arma::exp(y) %
           (arma::pow(arma::tanh(arma::exp(y)), 2) - 1);
     }
   }; // class TanhExpFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
