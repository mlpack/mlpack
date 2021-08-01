
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_sparse_coding_random_initializer.hpp:

Program Listing for File random_initializer.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_sparse_coding_random_initializer.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/sparse_coding/random_initializer.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SPARSE_CODING_RANDOM_INITIALIZER_HPP
   #define MLPACK_METHODS_SPARSE_CODING_RANDOM_INITIALIZER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace sparse_coding {
   
   class RandomInitializer
   {
    public:
     static void Initialize(const arma::mat& data,
                            const size_t atoms,
                            arma::mat& dictionary)
     {
       // Create random dictionary.
       dictionary.randn(data.n_rows, atoms);
   
       // Normalize each atom.
       for (size_t j = 0; j < atoms; ++j)
         dictionary.col(j) /= norm(dictionary.col(j), 2);
     }
   };
   
   } // namespace sparse_coding
   } // namespace mlpack
   
   #endif
