
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_init_rules_average_init.hpp:

Program Listing for File average_init.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_init_rules_average_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/init_rules/average_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AMF_AVERAGE_INIT_HPP
   #define MLPACK_METHODS_AMF_AVERAGE_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace amf {
   
   class AverageInitialization
   {
    public:
     // Empty constructor required for the InitializeRule template
     AverageInitialization() { }
   
     template<typename MatType>
     inline static void Initialize(const MatType& V,
                                   const size_t r,
                                   arma::mat& W,
                                   arma::mat& H)
     {
       const size_t n = V.n_rows;
       const size_t m = V.n_cols;
   
       double avgV = 0;
       double min = DBL_MAX;
   
       // Iterate over all elements in the matrix (for sparse matrices, this only
       // iterates over nonzeros).
       for (typename MatType::const_row_col_iterator it = V.begin();
           it != V.end(); ++it)
       {
         avgV += *it;
         // Track the minimum value.
         if (*it < min)
           min = *it;
       }
   
       avgV = sqrt(((avgV / (n * m)) - min) / r);
   
       // Initialize to random values.
       W.randu(n, r);
       H.randu(r, m);
   
       W += avgV;
       H += + avgV;
     }
   
     template<typename MatType>
     inline static void InitializeOne(const MatType& V,
                                      const size_t r,
                                      arma::mat& M,
                                      const bool whichMatrix = true)
     {
       const size_t n = V.n_rows;
       const size_t m = V.n_cols;
   
       double avgV = 0;
       double min = DBL_MAX;
   
       // Iterate over all elements in the matrix (for sparse matrices, this only
       // iterates over nonzeros).
       for (typename MatType::const_row_col_iterator it = V.begin();
           it != V.end(); ++it)
       {
         avgV += *it;
         // Track the minimum value.
         if (*it < min)
           min = *it;
       }
       if (whichMatrix)
       {
         // Initialize W to random values
         M.randu(n, r);
       }
       else
       {
         // Initialize H to random values
         M.randu(r, m);
       }
       M += sqrt(((avgV / (n * m)) - min) / r);
     }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace amf
   } // namespace mlpack
   
   #endif
