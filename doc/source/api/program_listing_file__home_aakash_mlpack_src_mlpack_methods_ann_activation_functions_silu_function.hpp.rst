
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_silu_function.hpp:

Program Listing for File silu_function.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_silu_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/silu_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

      title = {Sigmoid-Weighted Linear Units for Neural Network Function
               Approximation in Reinforcement Learning},
      author = {Stefan Elfwing and Eiji Uchibe and Kenji Doya},
      year = {2017},
      url = {https://arxiv.org/pdf/1702.03118.pdf},
      eprint = {1702.03118},
      archivePrefix = {arXiv},
      primaryClass = {cs.LG} }
   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SILU_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SILU_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann /* Artificial Neural Network */ {
   
   class SILUFunction
   {
    public:
     static double Fn(const double x)
     {
       return x / (1.0 + std::exp(-x));
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType &x, OutputVecType &y)
     {
       y = x / (1.0 + arma::exp(-x));
     }
   
     static double Deriv(const double x)
     {
       double sigmoid = 1.0 / (1.0 + std::exp(-x));
       return sigmoid * (1.0 + x * (1.0 - sigmoid));
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType &x, OutputVecType &y)
     {
       OutputVecType sigmoid = 1.0 / (1.0 + arma::exp(-x));
       y = sigmoid % (1.0 + x % (1.0 - sigmoid));
     }
   }; // class SILUFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
