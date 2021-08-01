
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_identity_function.hpp:

Program Listing for File identity_function.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_activation_functions_identity_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/activation_functions/identity_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_IDENTITY_FUNCTION_HPP
   #define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_IDENTITY_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class IdentityFunction
   {
    public:
     static double Fn(const double x)
     {
       return x;
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Fn(const InputVecType& x, OutputVecType& y)
     {
       y = x;
     }
   
     static double Deriv(const double /* x */)
     {
       return 1.0;
     }
   
     template<typename InputVecType, typename OutputVecType>
     static void Deriv(const InputVecType& y, OutputVecType& x)
     {
       x.ones(arma::size(y));
     }
   
     template<typename eT>
     static void Deriv(const arma::Cube<eT>& y, arma::Cube<eT>& x)
     {
       x.ones(y.n_rows, y.n_cols, y.n_slices);
     }
   }; // class IdentityFunction
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
