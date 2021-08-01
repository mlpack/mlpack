
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_logistic_function.hpp:

Program Listing for File logistic_function.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_logistic_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/logistic_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_LOGISTIC_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_LOGISTIC_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class LogisticFunction
   {
    public:
     template<typename eT>
     static double Fn(const eT x)
     {
       if (x < arma::Datum<eT>::log_max)
       {
         if (x > -arma::Datum<eT>::log_max)
           return 1.0 / (1.0 + std::exp(-x));
   
         return 0.0;
       }
   
       return 1.0;
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = (1.0 / (1 + arma::exp(-x)));
     }
   
     static double Deriv(const double x)
     {
       return x * (1.0 - x);
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& y, OutputVecType& x)
     {
       x = y % (1.0 - y);
     }
   
     static double Inv(const double y)
     {
       return arma::trunc_log(y / (1 - y));
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Inv(const InputVecType& y, OutputVecType& x)
     {
       x = arma::trunc_log(y / (1 - y));
     }
   }; // class LogisticFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
