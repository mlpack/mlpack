
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_space_split_projection_vector.hpp:

Program Listing for File projection_vector.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_space_split_projection_vector.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/space_split/projection_vector.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_SPILL_TREE_PROJECTION_VECTOR_HPP
   #define MLPACK_CORE_TREE_SPILL_TREE_PROJECTION_VECTOR_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "../bounds.hpp"
   namespace mlpack {
   namespace tree {
   
   class AxisParallelProjVector
   {
     size_t dim;
   
    public:
     AxisParallelProjVector(size_t dim = 0) :
         dim(dim)
     {};
   
     template<typename VecType>
     double Project(const VecType& point,
                    typename std::enable_if_t<IsVector<VecType>::value>* = 0) const
     {
       return point[dim];
     };
   
     template<typename MetricType, typename ElemType>
     math::RangeType<ElemType> Project(
         const bound::HRectBound<MetricType, ElemType>& bound) const
     {
       return bound[dim];
     };
   
     template<typename MetricType, typename VecType>
     math::RangeType<typename VecType::elem_type> Project(
         const bound::BallBound<MetricType, VecType>& bound) const
     {
       return bound[dim];
     };
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(dim));
     };
   };
   
   class ProjVector
   {
     arma::vec projVect;
   
    public:
     ProjVector() :
         projVect()
     {};
   
     ProjVector(const arma::vec& vect) :
         projVect(arma::normalise(vect))
     {};
   
     template<typename VecType>
     double Project(const VecType& point,
                    typename std::enable_if_t<IsVector<VecType>::value>* = 0) const
     {
       return arma::dot(point, projVect);
     };
   
     template<typename MetricType, typename VecType>
     math::RangeType<typename VecType::elem_type> Project(
         const bound::BallBound<MetricType, VecType>& bound) const
     {
       typedef typename VecType::elem_type ElemType;
       const double center = Project(bound.Center());
       const ElemType radius = bound.Radius();
       return math::RangeType<ElemType>(center - radius, center + radius);
     };
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(projVect));
     };
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
