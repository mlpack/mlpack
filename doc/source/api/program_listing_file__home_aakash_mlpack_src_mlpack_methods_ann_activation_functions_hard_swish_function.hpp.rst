
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_hard_swish_function.hpp:

Program Listing for File hard_swish_function.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_hard_swish_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/hard_swish_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author = {Howard A, Sandler M, Chu G, Chen LC, Chen B, Tan M, Wang W,
              Zhu Y, Pang R, Vasudevan V and Le QV},
     title = {Searching for MobileNetV3},
     year = {2019}
   }
   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_HARD_SWISH_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_HARD_SWISH_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   class HardSwishFunction
   {
    public:
     static double Fn(const double x)
     {
       if (x <= -3)
         return 0;
       else if (x >= 3)
         return x;
   
       return x * (x + 3) / 6;
     }
   
     template <typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType &x, OutputVecType &y)
     {
       y.set_size(size(x));
   
       for (size_t i = 0; i < x.n_elem; i++)
         y(i) = Fn(x(i));
     }
   
     static double Deriv(const double y)
     {
       if (y <= -3)
         return 0;
       else if (y >= 3)
         return 1;
   
       return (2 * y + 3.0) / 6.0;
     }
   
     template <typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType &y, OutputVecType &x)
     {
       x.set_size(size(y));
   
       for (size_t i = 0; i < y.n_elem; i++)
         x(i) = Deriv(y(i));
     }
   }; // class HardSwishFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
