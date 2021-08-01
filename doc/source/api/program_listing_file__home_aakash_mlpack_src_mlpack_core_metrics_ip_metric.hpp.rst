
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_metrics_ip_metric.hpp:

Program Listing for File ip_metric.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_metrics_ip_metric.hpp>` (``/home/aakash/mlpack/src/mlpack/core/metrics/ip_metric.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_FASTMKS_IP_METRIC_HPP
   #define MLPACK_METHODS_FASTMKS_IP_METRIC_HPP
   
   namespace mlpack {
   namespace metric {
   
   template<typename KernelType>
   class IPMetric
   {
    public:
     IPMetric();
   
     IPMetric(KernelType& kernel);
   
     ~IPMetric();
   
     IPMetric(const IPMetric& other);
   
     IPMetric& operator=(const IPMetric& other);
   
     template<typename VecTypeA, typename VecTypeB>
     typename VecTypeA::elem_type Evaluate(const VecTypeA& a, const VecTypeB& b);
   
     const KernelType& Kernel() const { return *kernel; }
     KernelType& Kernel() { return *kernel; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     KernelType* kernel;
     bool kernelOwner;
   };
   
   } // namespace metric
   } // namespace mlpack
   
   // Include implementation.
   #include "ip_metric_impl.hpp"
   
   #endif
