
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_make_alias.hpp:

Program Listing for File make_alias.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_make_alias.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/make_alias.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_MATH_MAKE_ALIAS_HPP
   #define MLPACK_CORE_MATH_MAKE_ALIAS_HPP
   
   namespace mlpack {
   namespace math {
   
   template<typename ElemType>
   arma::Cube<ElemType> MakeAlias(arma::Cube<ElemType>& input,
                                  const bool strict = true)
   {
     // Use the advanced constructor.
     return arma::Cube<ElemType>(input.memptr(), input.n_rows, input.n_cols,
         input.n_slices, false, strict);
   }
   
   template<typename ElemType>
   arma::Mat<ElemType> MakeAlias(arma::Mat<ElemType>& input,
                                 const bool strict = true)
   {
     // Use the advanced constructor.
     return arma::Mat<ElemType>(input.memptr(), input.n_rows, input.n_cols, false,
         strict);
   }
   
   template<typename ElemType>
   arma::Row<ElemType> MakeAlias(arma::Row<ElemType>& input,
                                 const bool strict = true)
   {
     // Use the advanced constructor.
     return arma::Row<ElemType>(input.memptr(), input.n_elem, false, strict);
   }
   
   template<typename ElemType>
   arma::Col<ElemType> MakeAlias(arma::Col<ElemType>& input,
                                 const bool strict = true)
   {
     // Use the advanced constructor.
     return arma::Col<ElemType>(input.memptr(), input.n_elem, false, strict);
   }
   
   template<typename ElemType>
   arma::SpMat<ElemType> MakeAlias(const arma::SpMat<ElemType>& input,
                                   const bool /* strict */ = true)
   {
     // Make a copy...
     return arma::SpMat<ElemType>(input);
   }
   
   template<typename ElemType>
   arma::SpRow<ElemType> MakeAlias(const arma::SpRow<ElemType>& input,
                                   const bool /* strict */ = true)
   {
     // Make a copy...
     return arma::SpRow<ElemType>(input);
   }
   
   template<typename ElemType>
   arma::SpCol<ElemType> MakeAlias(const arma::SpCol<ElemType>& input,
                                   const bool /* strict */ = true)
   {
     // Make a copy...
     return arma::SpCol<ElemType>(input);
   }
   
   template<typename ElemType>
   void ClearAlias(arma::Mat<ElemType>& mat)
   {
     if (mat.mem_state >= 1)
       mat.reset();
   }
   
   template<typename ElemType>
   void ClearAlias(arma::SpMat<ElemType>& /* mat */)
   {
     // Nothing to do.
   }
   
   
   } // namespace math
   } // namespace mlpack
   
   #endif
