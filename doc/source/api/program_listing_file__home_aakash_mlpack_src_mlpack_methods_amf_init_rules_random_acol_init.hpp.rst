
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_init_rules_random_acol_init.hpp:

Program Listing for File random_acol_init.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_init_rules_random_acol_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/init_rules/random_acol_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LMF_RANDOM_ACOL_INIT_HPP
   #define MLPACK_METHODS_LMF_RANDOM_ACOL_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/random.hpp>
   
   namespace mlpack {
   namespace amf {
   
   template<size_t columnsToAverage = 5>
   class RandomAcolInitialization
   {
    public:
     // Empty constructor required for the InitializeRule template
     RandomAcolInitialization()
     { }
   
     template<typename MatType>
     inline static void Initialize(const MatType& V,
                                   const size_t r,
                                   arma::mat& W,
                                   arma::mat& H)
     {
       const size_t n = V.n_rows;
       const size_t m = V.n_cols;
   
       if (columnsToAverage > m)
       {
         Log::Warn << "Number of random columns (columnsToAverage) is more than "
             << "the number of columns available in the V matrix; weird results "
             << "may ensue!" << std::endl;
       }
   
       W.zeros(n, r);
   
       // Initialize W matrix with random columns.
       for (size_t col = 0; col < r; col++)
       {
         for (size_t randCol = 0; randCol < columnsToAverage; randCol++)
         {
           // .col() does not work in this case, as of Armadillo 3.920.
           W.unsafe_col(col) += V.col(math::RandInt(0, m));
         }
       }
   
       // Now divide by p.
       W /= columnsToAverage;
   
       // Initialize H to random values.
       H.randu(r, m);
     }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace amf
   } // namespace mlpack
   
   #endif
