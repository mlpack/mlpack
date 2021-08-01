
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_arma_traits.hpp:

Program Listing for File arma_traits.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_arma_traits.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/arma_traits.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_ARMA_TRAITS_HPP
   #define MLPACK_CORE_UTIL_ARMA_TRAITS_HPP
   
   // Structs have public members by default (that's why they are chosen over
   // classes).
   
   template<typename VecType>
   struct IsVector
   {
     const static bool value = false;
   };
   
   // Commenting out the first template per case, because
   // Visual Studio doesn't like this instantiaion pattern (error C2910).
   // template<>
   template<typename eT>
   struct IsVector<arma::Col<eT> >
   {
     const static bool value = true;
   };
   
   // template<>
   template<typename eT>
   struct IsVector<arma::SpCol<eT> >
   {
     const static bool value = true;
   };
   
   // template<>
   template<typename eT>
   struct IsVector<arma::Row<eT> >
   {
     const static bool value = true;
   };
   
   // template<>
   template<typename eT>
   struct IsVector<arma::SpRow<eT> >
   {
     const static bool value = true;
   };
   
   // template<>
   template<typename eT>
   struct IsVector<arma::subview_col<eT> >
   {
     const static bool value = true;
   };
   
   // template<>
   template<typename eT>
   struct IsVector<arma::subview_row<eT> >
   {
     const static bool value = true;
   };
   
   
   #if ((ARMA_VERSION_MAJOR >= 10) || \
       ((ARMA_VERSION_MAJOR == 9) && (ARMA_VERSION_MINOR >= 869)))
   
     // Armadillo 9.869+ has SpSubview_col and SpSubview_row
   
     template<typename eT>
     struct IsVector<arma::SpSubview_col<eT> >
     {
       const static bool value = true;
     };
   
     template<typename eT>
     struct IsVector<arma::SpSubview_row<eT> >
     {
       const static bool value = true;
     };
   
   #else
   
     // fallback for older Armadillo versions
   
     template<typename eT>
     struct IsVector<arma::SpSubview<eT> >
     {
       const static bool value = true;
     };
   
   #endif
   
   #endif
