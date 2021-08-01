
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_softplus_function.hpp:

Program Listing for File softplus_function.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_softplus_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/softplus_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author    = {Dugas, Charles and Bengio, Yoshua and Belisle, Francois and
                  Nadeau, Claude and Garcia, Rene},
     title     = {Incorporating Second-Order Functional Knowledge for Better
                  Option Pricing},
     booktitle = {Advances in Neural Information Processing Systems},
     year      = {2001}
   }
   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFTPLUS_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFTPLUS_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class SoftplusFunction
   {
    public:
     static double Fn(const double x)
     {
       const double val = std::log(1 + std::exp(x));
       if (std::isfinite(val))
         return val;
       return x;
     }
   
     template<typename InputType, typename OutputType>
     static void Fn(const InputType& x, OutputType& y)
     {
       y.set_size(arma::size(x));
   
       for (size_t i = 0; i < x.n_elem; ++i)
         y(i) = Fn(x(i));
     }
   
     static double Deriv(const double y)
     {
       return 1.0 / (1 + std::exp(-y));
     }
   
     template<typename InputType, typename OutputType>
     static void Deriv(const InputType& y, OutputType& x)
     {
       x = 1.0 / (1 + arma::exp(-y));
     }
   
     static double Inv(const double y)
     {
       const double val = std::log(std::exp(y) - 1);
       if (std::isfinite(val))
         return val;
       return y;
     }
   
     template<typename InputType, typename OutputType>
     static void Inv(const InputType& y, OutputType& x)
     {
       x.set_size(arma::size(y));
   
       for (size_t i = 0; i < y.n_elem; ++i)
         x(i) = Inv(y(i));
     }
   }; // class SoftplusFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
