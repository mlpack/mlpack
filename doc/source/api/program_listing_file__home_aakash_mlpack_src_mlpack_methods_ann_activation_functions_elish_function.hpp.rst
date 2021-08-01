
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_elish_function.hpp:

Program Listing for File elish_function.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_elish_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/elish_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

      title = {The Quest for the Golden Activation Function},
      author = {Mina Basirat and Peter M. Roth},
      year = {2018},
      url = {https://arxiv.org/pdf/1808.00783.pdf},
      eprint = {1808.00783},
      archivePrefix = {arXiv},
      primaryClass = {cs.NE} }
   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ELISH_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ELISH_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class ElishFunction
   {
    public:
     static double Fn(const double x)
     {
       if (x < 0.0)
         return (std::exp(x) - 1) / (1 + std::exp(-x));
   
       return x / (1 + std::exp(-x));
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = ((x < 0.0) % ((arma::exp(x) -1) / (1 + arma::exp(-x))))
           + ((x >= 0.0) % (x / (1 + arma::exp(-x))));
     }
   
     static double Deriv(const double y)
     {
       if (y < 0.0)
       {
         return std::exp(y) - 2 / (1 + std::exp(y)) +
             2 / std::pow(1 + std::exp(y) , 2);
       }
   
       return 1 / (1 + std::exp(-y)) + y * std::exp(-y) /
           std::pow(1 + std::exp(-y) , 2);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& y, OutputVecType& x)
     {
       x = ((y < 0.0) % (arma::exp(y) - 2 / (1 + arma::exp(y)) + 2 / arma::pow(
           1 + arma::exp(y), 2))) + ((y >= 0.0) % (1 / (1 + arma::exp(-y)) + y %
           arma::exp(-y) / arma::pow(1 + arma::exp(-y), 2)));
     }
   }; // class ElishFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
