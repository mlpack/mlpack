
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_hard_sigmoid_function.hpp:

Program Listing for File hard_sigmoid_function.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_hard_sigmoid_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/hard_sigmoid_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_HARD_SIGMOID_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_HARD_SIGMOID_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <algorithm>
   
   namespace mlpack {
   namespace ann  {
   
   class HardSigmoidFunction
   {
    public:
     static double Fn(const double x)
     {
       return std::min(1.0, std::max(0.0, 0.2 * x + 0.5));
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y.set_size(size(x));
   
       for (size_t i = 0; i < x.n_elem; ++i)
         y(i) = Fn(x(i));
     }
   
     static double Deriv(const double y)
     {
       if (y == 0.0 || y == 1.0)
       {
         return 0.0;
       }
       return 0.2;
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& y, OutputVecType& x)
     {
       x.set_size(size(y));
   
       for (size_t i = 0; i < y.n_elem; ++i)
       {
         x(i) = Deriv(y(i));
       }
     }
   }; // class HardSigmoidFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
