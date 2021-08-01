
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_metrics_lmetric.hpp:

Program Listing for File lmetric.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_metrics_lmetric.hpp>` (``/home/aakash/mlpack/src/mlpack/core/metrics/lmetric.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_METRICS_LMETRIC_HPP
   #define MLPACK_CORE_METRICS_LMETRIC_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace metric {
   
   template<int TPower, bool TTakeRoot = true>
   class LMetric
   {
    public:
     LMetric() { }
   
     template<typename VecTypeA, typename VecTypeB>
     static typename VecTypeA::elem_type Evaluate(const VecTypeA& a,
                                                  const VecTypeB& b);
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   
     static const int Power = TPower;
     static const bool TakeRoot = TTakeRoot;
   };
   
   // Convenience typedefs.
   
   typedef LMetric<1, false> ManhattanDistance;
   
   typedef LMetric<2, false> SquaredEuclideanDistance;
   
   typedef LMetric<2, true> EuclideanDistance;
   
   typedef LMetric<INT_MAX, false> ChebyshevDistance;
   
   
   } // namespace metric
   } // namespace mlpack
   
   // Include implementation.
   #include "lmetric_impl.hpp"
   
   #endif
