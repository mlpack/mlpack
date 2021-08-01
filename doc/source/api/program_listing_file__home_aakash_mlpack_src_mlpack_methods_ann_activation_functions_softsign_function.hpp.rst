
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_softsign_function.hpp:

Program Listing for File softsign_function.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_softsign_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/softsign_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     title={title={Understanding the difficulty of training deep feedforward
     neural networks},
     author={Glorot, Xavier and Bengio, Yoshua},
     booktitle={Proceedings of AISTATS 2010},
     year={2010}
   }
   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFTSIGN_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFTSIGN_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class SoftsignFunction
   {
    public:
     static double Fn(const double x)
     {
       if (x < DBL_MAX)
         return x > -DBL_MAX ? x / (1.0 + std::abs(x)) : -1.0;
       return 1.0;
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y.set_size(arma::size(x));
   
       for (size_t i = 0; i < x.n_elem; ++i)
         y(i) = Fn(x(i));
     }
   
     static double Deriv(const double y)
     {
       return std::pow(1.0 - std::abs(y), 2);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& y, OutputVecType& x)
     {
       x = arma::pow(1.0 - arma::abs(y), 2);
     }
   
     static double Inv(const double y)
     {
       if (y > 0)
         return y < 1 ? -y / (y - 1) : DBL_MAX;
       else
         return y > -1 ? y / (1 + y) : -DBL_MAX;
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Inv(const InputVecType& y, OutputVecType& x)
     {
       x.set_size(arma::size(y));
   
       for (size_t i = 0; i < y.n_elem; ++i)
         x(i) = Inv(y(i));
     }
   }; // class SoftsignFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
