
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_hollow_ball_bound.hpp:

Program Listing for File hollow_ball_bound.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_hollow_ball_bound.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/hollow_ball_bound.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_HPP
   #define MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include "bound_traits.hpp"
   
   namespace mlpack {
   namespace bound {
   
   template<typename TMetricType = metric::LMetric<2, true>,
            typename ElemType = double>
   class HollowBallBound
   {
    public:
     typedef TMetricType MetricType;
   
    private:
     math::RangeType<ElemType> radii;
     arma::Col<ElemType> center;
     arma::Col<ElemType> hollowCenter;
     MetricType* metric;
   
     bool ownsMetric;
   
    public:
     HollowBallBound();
   
     HollowBallBound(const size_t dimension);
   
     template<typename VecType>
     HollowBallBound(const ElemType innerRadius,
                     const ElemType outerRadius,
                     const VecType& center);
   
     HollowBallBound(const HollowBallBound& other);
   
     HollowBallBound& operator=(const HollowBallBound& other);
   
     HollowBallBound(HollowBallBound&& other);
   
     HollowBallBound& operator=(HollowBallBound&& other);
   
     ~HollowBallBound();
   
     ElemType OuterRadius() const { return radii.Hi(); }
     ElemType& OuterRadius() { return radii.Hi(); }
   
     ElemType InnerRadius() const { return radii.Lo(); }
     ElemType& InnerRadius() { return radii.Lo(); }
   
     const arma::Col<ElemType>& Center() const { return center; }
     arma::Col<ElemType>& Center() { return center; }
   
     const arma::Col<ElemType>& HollowCenter() const { return hollowCenter; }
     arma::Col<ElemType>& HollowCenter() { return hollowCenter; }
   
     size_t Dim() const { return center.n_elem; }
   
     ElemType MinWidth() const { return radii.Hi() * 2.0; }
   
     math::RangeType<ElemType> operator[](const size_t i) const;
   
     template<typename VecType>
     bool Contains(const VecType& point) const;
   
     bool Contains(const HollowBallBound& other) const;
   
     template<typename VecType>
     void Center(VecType& center) const { center = this->center; }
   
     template<typename VecType>
     ElemType MinDistance(const VecType& point,
                          typename std::enable_if_t<IsVector<VecType>::value>* = 0)
         const;
   
     ElemType MinDistance(const HollowBallBound& other) const;
   
     template<typename VecType>
     ElemType MaxDistance(const VecType& point,
                          typename std::enable_if_t<IsVector<VecType>::value>* = 0)
         const;
   
     ElemType MaxDistance(const HollowBallBound& other) const;
   
     template<typename VecType>
     math::RangeType<ElemType> RangeDistance(
         const VecType& other,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
   
     math::RangeType<ElemType> RangeDistance(const HollowBallBound& other) const;
   
     template<typename MatType>
     const HollowBallBound& operator|=(const MatType& data);
   
     const HollowBallBound& operator|=(const HollowBallBound& other);
   
     ElemType Diameter() const { return 2 * radii.Hi(); }
   
     const MetricType& Metric() const { return *metric; }
     MetricType& Metric() { return *metric; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   };
   
   template<typename MetricType, typename ElemType>
   struct BoundTraits<HollowBallBound<MetricType, ElemType>>
   {
     const static bool HasTightBounds = false;
   };
   
   } // namespace bound
   } // namespace mlpack
   
   #include "hollow_ball_bound_impl.hpp"
   
   #endif // MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_HPP
