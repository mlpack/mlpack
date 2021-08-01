
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_metrics_iou_metric.hpp:

Program Listing for File iou_metric.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_metrics_iou_metric.hpp>` (``/home/aakash/mlpack/src/mlpack/core/metrics/iou_metric.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_METRICS_IOU_HPP
   #define MLPACK_CORE_METRICS_IOU_HPP
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace metric {
   
   template<bool UseCoordinates = false>
   class IoU
   {
    public:
     IoU()
     {
       // Nothing to do here.
     }
   
     template<typename VecTypeA, typename VecTypeB>
     static typename VecTypeA::elem_type Evaluate(const VecTypeA& a,
                                                  const VecTypeB& b);
   
     static const bool useCoordinates = UseCoordinates;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   }; // class IoU
   
   } // namespace metric
   } // namespace mlpack
   
   // Include implementation.
   #include "iou_metric_impl.hpp"
   
   #endif
