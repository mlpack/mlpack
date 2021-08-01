
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_space_split_hyperplane.hpp:

Program Listing for File hyperplane.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_space_split_hyperplane.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/space_split/hyperplane.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_SPILL_TREE_HYPERPLANE_HPP
   #define MLPACK_CORE_TREE_SPILL_TREE_HYPERPLANE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "projection_vector.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename BoundT, typename ProjVectorT>
   class HyperplaneBase
   {
    public:
     typedef BoundT BoundType;
     typedef ProjVectorT ProjVectorType;
   
    private:
     ProjVectorType projVect;
   
     double splitVal;
   
    public:
     HyperplaneBase() :
         splitVal(DBL_MAX)
     {};
   
     HyperplaneBase(const ProjVectorType& projVect, double splitVal) :
         projVect(projVect),
         splitVal(splitVal)
     {};
   
     template<typename VecType>
     double Project(const VecType& point,
                    typename std::enable_if_t<IsVector<VecType>::value>* = 0) const
     {
       if (splitVal == DBL_MAX)
         return 0;
       return projVect.Project(point) - splitVal;
     };
   
     template<typename VecType>
     bool Left(const VecType& point,
               typename std::enable_if_t<IsVector<VecType>::value>* = 0) const
     {
       return Project(point) <= 0;
     };
   
     template<typename VecType>
     bool Right(const VecType& point,
               typename std::enable_if_t<IsVector<VecType>::value>* = 0) const
     {
       return Project(point) > 0;
     };
   
     bool Left(const BoundType& bound) const
     {
       if (splitVal == DBL_MAX)
         return true;
       return projVect.Project(bound).Hi() <= splitVal;
     };
   
     bool Right(const BoundType& bound) const
     {
       if (splitVal == DBL_MAX)
         return false;
       return projVect.Project(bound).Lo() > splitVal;
     };
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(projVect));
       ar(CEREAL_NVP(splitVal));
     };
   };
   
   template<typename MetricType>
   using AxisOrthogonalHyperplane = HyperplaneBase<bound::HRectBound<MetricType>,
       AxisParallelProjVector>;
   
   template<typename MetricType>
   using Hyperplane = HyperplaneBase<bound::BallBound<MetricType>, ProjVector>;
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
