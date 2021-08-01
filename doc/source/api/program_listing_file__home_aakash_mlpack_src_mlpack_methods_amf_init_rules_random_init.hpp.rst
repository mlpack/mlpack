
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_init_rules_random_init.hpp:

Program Listing for File random_init.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_init_rules_random_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/init_rules/random_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LMF_RANDOM_INIT_HPP
   #define MLPACK_METHODS_LMF_RANDOM_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace amf {
   
   class RandomInitialization
   {
    public:
     // Empty constructor required for the InitializeRule template
     RandomInitialization() { }
   
     template<typename MatType>
     inline static void Initialize(const MatType& V,
                                   const size_t r,
                                   arma::mat& W,
                                   arma::mat& H)
     {
       // Simple implementation (left in the header file due to its simplicity).
       const size_t n = V.n_rows;
       const size_t m = V.n_cols;
   
       // Initialize to random values.
       W.randu(n, r);
       H.randu(r, m);
     }
   
     template<typename MatType>
     inline void InitializeOne(const MatType& V,
                               const size_t r,
                               arma::mat& M,
                               const bool whichMatrix = true)
     {
       // Simple implementation (left in the header file due to its simplicity).
       const size_t n = V.n_rows;
       const size_t m = V.n_cols;
   
       // Initialize W or H to random values
       if (whichMatrix)
       {
         M.randu(n, r);
       }
       else
       {
         M.randu(r, m);
       }
     }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace amf
   } // namespace mlpack
   
   #endif
