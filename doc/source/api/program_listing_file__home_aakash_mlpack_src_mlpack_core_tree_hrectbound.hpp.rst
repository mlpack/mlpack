
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_hrectbound.hpp:

Program Listing for File hrectbound.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_hrectbound.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/hrectbound.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_HRECTBOUND_HPP
   #define MLPACK_CORE_TREE_HRECTBOUND_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/range.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include "bound_traits.hpp"
   
   namespace mlpack {
   namespace bound {
   
   namespace meta  {
   
   template<typename MetricType>
   struct IsLMetric
   {
     static const bool Value = false;
   };
   
   template<int Power, bool TakeRoot>
   struct IsLMetric<metric::LMetric<Power, TakeRoot>>
   {
     static const bool Value = true;
   };
   
   } // namespace meta
   
   template<typename MetricType = metric::LMetric<2, true>,
            typename ElemType = double>
   class HRectBound
   {
     // It is required that HRectBound have an LMetric as the given MetricType.
     static_assert(meta::IsLMetric<MetricType>::Value == true,
         "HRectBound can only be used with the LMetric<> metric type.");
   
    public:
     HRectBound();
   
     HRectBound(const size_t dimension);
   
     HRectBound(const HRectBound& other);
   
     HRectBound& operator=(const HRectBound& other);
   
     HRectBound(HRectBound&& other);
   
     HRectBound& operator=(HRectBound&& other);
   
     ~HRectBound();
   
     void Clear();
   
     size_t Dim() const { return dim; }
   
     math::RangeType<ElemType>& operator[](const size_t i) { return bounds[i]; }
     const math::RangeType<ElemType>& operator[](const size_t i) const
     { return bounds[i]; }
   
     ElemType MinWidth() const { return minWidth; }
     ElemType& MinWidth() { return minWidth; }
   
     const MetricType& Metric() const { return metric; }
     MetricType& Metric() { return metric; }
   
     void Center(arma::Col<ElemType>& center) const;
   
     ElemType Volume() const;
   
     template<typename VecType>
     ElemType MinDistance(const VecType& point,
                          typename std::enable_if_t<IsVector<VecType>::value>* = 0)
         const;
   
     ElemType MinDistance(const HRectBound& other) const;
   
     template<typename VecType>
     ElemType MaxDistance(const VecType& point,
                          typename std::enable_if_t<IsVector<VecType>::value>* = 0)
         const;
   
     ElemType MaxDistance(const HRectBound& other) const;
   
     math::RangeType<ElemType> RangeDistance(const HRectBound& other) const;
   
     template<typename VecType>
     math::RangeType<ElemType> RangeDistance(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
   
     template<typename MatType>
     HRectBound& operator|=(const MatType& data);
   
     HRectBound& operator|=(const HRectBound& other);
   
     template<typename VecType>
     bool Contains(const VecType& point) const;
   
     bool Contains(const HRectBound& bound) const;
   
     HRectBound operator&(const HRectBound& bound) const;
   
     HRectBound& operator&=(const HRectBound& bound);
   
     ElemType Overlap(const HRectBound& bound) const;
   
     ElemType Diameter() const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     size_t dim;
     math::RangeType<ElemType>* bounds;
     ElemType minWidth;
     MetricType metric;
   };
   
   // A specialization of BoundTraits for this class.
   template<typename MetricType, typename ElemType>
   struct BoundTraits<HRectBound<MetricType, ElemType>>
   {
     const static bool HasTightBounds = true;
   };
   
   } // namespace bound
   } // namespace mlpack
   
   #include "hrectbound_impl.hpp"
   
   #endif // MLPACK_CORE_TREE_HRECTBOUND_HPP
