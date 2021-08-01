
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_sparse_coding_nothing_initializer.hpp:

Program Listing for File nothing_initializer.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_sparse_coding_nothing_initializer.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/sparse_coding/nothing_initializer.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SPARSE_CODING_NOTHING_INITIALIZER_HPP
   #define MLPACK_METHODS_SPARSE_CODING_NOTHING_INITIALIZER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace sparse_coding {
   
   class NothingInitializer
   {
    public:
     static void Initialize(const arma::mat& /* data */,
                            const size_t /* atoms */,
                            arma::mat& /* dictionary */)
     {
       // Do nothing!
     }
   };
   
   } // namespace sparse_coding
   } // namespace mlpack
   
   #endif
