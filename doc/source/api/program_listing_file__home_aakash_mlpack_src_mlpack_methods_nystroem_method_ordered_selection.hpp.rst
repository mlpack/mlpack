
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_nystroem_method_ordered_selection.hpp:

Program Listing for File ordered_selection.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_nystroem_method_ordered_selection.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/nystroem_method/ordered_selection.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NYSTROEM_METHOD_ORDERED_SELECTION_HPP
   #define MLPACK_METHODS_NYSTROEM_METHOD_ORDERED_SELECTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class OrderedSelection
   {
    public:
     const static arma::Col<size_t> Select(const arma::mat& /* data */,
                                           const size_t m)
     {
       // This generates [0 1 2 3 ... (m - 1)].
       return arma::linspace<arma::Col<size_t> >(0, m - 1, m);
     }
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
