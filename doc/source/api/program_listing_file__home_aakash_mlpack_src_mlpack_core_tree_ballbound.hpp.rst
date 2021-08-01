
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_ballbound.hpp:

Program Listing for File ballbound.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_ballbound.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/ballbound.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_BALLBOUND_HPP
   #define MLPACK_CORE_TREE_BALLBOUND_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include "bound_traits.hpp"
   
   namespace mlpack {
   namespace bound {
   
   template<typename MetricType = metric::LMetric<2, true>,
            typename VecType = arma::vec>
   class BallBound
   {
    public:
     typedef typename VecType::elem_type ElemType;
     typedef VecType Vec;
   
    private:
     ElemType radius;
     VecType center;
     MetricType* metric;
   
     bool ownsMetric;
   
    public:
     BallBound();
   
     BallBound(const size_t dimension);
   
     BallBound(const ElemType radius, const VecType& center);
   
     BallBound(const BallBound& other);
   
     BallBound& operator=(const BallBound& other);
   
     BallBound(BallBound&& other);
   
     BallBound& operator=(BallBound&& other);
   
     ~BallBound();
   
     ElemType Radius() const { return radius; }
     ElemType& Radius() { return radius; }
   
     const VecType& Center() const { return center; }
     VecType& Center() { return center; }
   
     size_t Dim() const { return center.n_elem; }
   
     ElemType MinWidth() const { return radius * 2.0; }
   
     math::RangeType<ElemType> operator[](const size_t i) const;
   
     bool Contains(const VecType& point) const;
   
     void Center(VecType& center) const { center = this->center; }
   
     template<typename OtherVecType>
     ElemType MinDistance(
         const OtherVecType& point,
         typename std::enable_if_t<IsVector<OtherVecType>::value>* = 0) const;
   
     ElemType MinDistance(const BallBound& other) const;
   
     template<typename OtherVecType>
     ElemType MaxDistance(
         const OtherVecType& point,
         typename std::enable_if_t<IsVector<OtherVecType>::value>* = 0) const;
   
     ElemType MaxDistance(const BallBound& other) const;
   
     template<typename OtherVecType>
     math::RangeType<ElemType> RangeDistance(
         const OtherVecType& other,
         typename std::enable_if_t<IsVector<OtherVecType>::value>* = 0) const;
   
     math::RangeType<ElemType> RangeDistance(const BallBound& other) const;
   
     const BallBound& operator|=(const BallBound& other);
   
     template<typename MatType>
     const BallBound& operator|=(const MatType& data);
   
     ElemType Diameter() const { return 2 * radius; }
   
     const MetricType& Metric() const { return *metric; }
     MetricType& Metric() { return *metric; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   };
   
   template<typename MetricType, typename VecType>
   struct BoundTraits<BallBound<MetricType, VecType>>
   {
     const static bool HasTightBounds = false;
   };
   
   } // namespace bound
   } // namespace mlpack
   
   #include "ballbound_impl.hpp"
   
   #endif // MLPACK_CORE_TREE_DBALLBOUND_HPP
