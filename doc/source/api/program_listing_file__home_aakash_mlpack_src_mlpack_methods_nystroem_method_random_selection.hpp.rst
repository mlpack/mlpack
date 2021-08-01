
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_nystroem_method_random_selection.hpp:

Program Listing for File random_selection.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_nystroem_method_random_selection.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/nystroem_method/random_selection.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NYSTROEM_METHOD_RANDOM_SELECTION_HPP
   #define MLPACK_METHODS_NYSTROEM_METHOD_RANDOM_SELECTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class RandomSelection
   {
    public:
     const static arma::Col<size_t> Select(const arma::mat& data, const size_t m)
     {
       arma::Col<size_t> selectedPoints(m);
       for (size_t i = 0; i < m; ++i)
         selectedPoints(i) = math::RandInt(0, data.n_cols);
   
       return selectedPoints;
     }
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
