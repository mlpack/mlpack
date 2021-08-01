
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_rann_ra_util.hpp:

Program Listing for File ra_util.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_rann_ra_util.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/rann/ra_util.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RANN_RA_UTIL_HPP
   #define MLPACK_METHODS_RANN_RA_UTIL_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace neighbor {
   
   class RAUtil
   {
    public:
     static size_t MinimumSamplesReqd(const size_t n,
                                      const size_t k,
                                      const double tau,
                                      const double alpha);
   
     static double SuccessProbability(const size_t n,
                                      const size_t k,
                                      const size_t m,
                                      const size_t t);
   
     static void ObtainDistinctSamples(const size_t numSamples,
                                       const size_t rangeUpperBound,
                                       arma::uvec& distinctSamples);
   };
   
   } // namespace neighbor
   } // namespace mlpack
   
   #endif
