
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_sparse_coding_data_dependent_random_initializer.hpp:

Program Listing for File data_dependent_random_initializer.hpp
==============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_sparse_coding_data_dependent_random_initializer.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/sparse_coding/data_dependent_random_initializer.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SPARSE_CODING_DATA_DEPENDENT_RANDOM_INITIALIZER_HPP
   #define MLPACK_METHODS_SPARSE_CODING_DATA_DEPENDENT_RANDOM_INITIALIZER_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/random.hpp>
   
   namespace mlpack {
   namespace sparse_coding {
   
   class DataDependentRandomInitializer
   {
    public:
     static void Initialize(const arma::mat& data,
                            const size_t atoms,
                            arma::mat& dictionary)
     {
       // Set the size of the dictionary.
       dictionary.set_size(data.n_rows, atoms);
   
       // Create each atom.
       for (size_t i = 0; i < atoms; ++i)
       {
         // Add three atoms together.
         dictionary.col(i) = (data.col(math::RandInt(data.n_cols)) +
             data.col(math::RandInt(data.n_cols)) +
             data.col(math::RandInt(data.n_cols)));
   
         // Now normalize the atom.
         dictionary.col(i) /= norm(dictionary.col(i), 2);
       }
     }
   };
   
   } // namespace sparse_coding
   } // namespace mlpack
   
   #endif
