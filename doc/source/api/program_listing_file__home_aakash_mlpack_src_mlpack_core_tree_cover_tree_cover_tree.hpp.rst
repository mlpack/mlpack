
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_cover_tree_cover_tree.hpp:

Program Listing for File cover_tree.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_cover_tree_cover_tree.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/cover_tree/cover_tree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_HPP
   #define MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/range.hpp>
   
   #include "../statistic.hpp"
   #include "first_point_is_root.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType = metric::LMetric<2, true>,
            typename StatisticType = EmptyStatistic,
            typename MatType = arma::mat,
            typename RootPointPolicy = FirstPointIsRoot>
   class CoverTree
   {
    public:
     typedef MatType Mat;
     typedef typename MatType::elem_type ElemType;
   
     CoverTree(const MatType& dataset,
               const ElemType base = 2.0,
               MetricType* metric = NULL);
   
     CoverTree(const MatType& dataset,
               MetricType& metric,
               const ElemType base = 2.0);
   
     CoverTree(MatType&& dataset,
               const ElemType base = 2.0);
   
     CoverTree(MatType&& dataset,
               MetricType& metric,
               const ElemType base = 2.0);
   
     CoverTree(const MatType& dataset,
               const ElemType base,
               const size_t pointIndex,
               const int scale,
               CoverTree* parent,
               const ElemType parentDistance,
               arma::Col<size_t>& indices,
               arma::vec& distances,
               size_t nearSetSize,
               size_t& farSetSize,
               size_t& usedSetSize,
               MetricType& metric = NULL);
   
     CoverTree(const MatType& dataset,
               const ElemType base,
               const size_t pointIndex,
               const int scale,
               CoverTree* parent,
               const ElemType parentDistance,
               const ElemType furthestDescendantDistance,
               MetricType* metric = NULL);
   
     CoverTree(const CoverTree& other);
   
     CoverTree(CoverTree&& other);
   
     CoverTree& operator=(const CoverTree& other);
   
     CoverTree& operator=(CoverTree&& other);
   
     template<typename Archive>
     CoverTree(
         Archive& ar,
         const typename std::enable_if_t<cereal::is_loading<Archive>()>* = 0);
   
     ~CoverTree();
   
     template<typename RuleType>
     class SingleTreeTraverser;
   
     template<typename RuleType>
     class DualTreeTraverser;
   
     template<typename RuleType>
     using BreadthFirstDualTreeTraverser = DualTreeTraverser<RuleType>;
   
     const MatType& Dataset() const { return *dataset; }
   
     size_t Point() const { return point; }
     size_t Point(const size_t) const { return point; }
   
     bool IsLeaf() const { return (children.size() == 0); }
     size_t NumPoints() const { return 1; }
   
     const CoverTree& Child(const size_t index) const { return *children[index]; }
     CoverTree& Child(const size_t index) { return *children[index]; }
   
     CoverTree*& ChildPtr(const size_t index) { return children[index]; }
   
     size_t NumChildren() const { return children.size(); }
   
     const std::vector<CoverTree*>& Children() const { return children; }
     std::vector<CoverTree*>& Children() { return children; }
   
     size_t NumDescendants() const;
   
     size_t Descendant(const size_t index) const;
   
     int Scale() const { return scale; }
     int& Scale() { return scale; }
   
     ElemType Base() const { return base; }
     ElemType& Base() { return base; }
   
     const StatisticType& Stat() const { return stat; }
     StatisticType& Stat() { return stat; }
   
     template<typename VecType>
     size_t GetNearestChild(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0);
   
     template<typename VecType>
     size_t GetFurthestChild(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0);
   
     size_t GetNearestChild(const CoverTree& queryNode);
   
     size_t GetFurthestChild(const CoverTree& queryNode);
   
     ElemType MinDistance(const CoverTree& other) const;
   
     ElemType MinDistance(const CoverTree& other, const ElemType distance) const;
   
     ElemType MinDistance(const arma::vec& other) const;
   
     ElemType MinDistance(const arma::vec& other, const ElemType distance) const;
   
     ElemType MaxDistance(const CoverTree& other) const;
   
     ElemType MaxDistance(const CoverTree& other, const ElemType distance) const;
   
     ElemType MaxDistance(const arma::vec& other) const;
   
     ElemType MaxDistance(const arma::vec& other, const ElemType distance) const;
   
     math::RangeType<ElemType> RangeDistance(const CoverTree& other) const;
   
     math::RangeType<ElemType> RangeDistance(const CoverTree& other,
                                             const ElemType distance) const;
   
     math::RangeType<ElemType> RangeDistance(const arma::vec& other) const;
   
     math::RangeType<ElemType> RangeDistance(const arma::vec& other,
                                             const ElemType distance) const;
   
     CoverTree* Parent() const { return parent; }
     CoverTree*& Parent() { return parent; }
   
     ElemType ParentDistance() const { return parentDistance; }
     ElemType& ParentDistance() { return parentDistance; }
   
     ElemType FurthestPointDistance() const { return 0.0; }
   
     ElemType FurthestDescendantDistance() const
     { return furthestDescendantDistance; }
     ElemType& FurthestDescendantDistance() { return furthestDescendantDistance; }
   
     ElemType MinimumBoundDistance() const { return furthestDescendantDistance; }
   
     void Center(arma::vec& center) const
     {
       center = arma::vec(dataset->col(point));
     }
   
     MetricType& Metric() const { return *metric; }
   
    private:
     const MatType* dataset;
     size_t point;
     std::vector<CoverTree*> children;
     int scale;
     ElemType base;
     StatisticType stat;
     size_t numDescendants;
     CoverTree* parent;
     ElemType parentDistance;
     ElemType furthestDescendantDistance;
     bool localMetric;
     bool localDataset;
     MetricType* metric;
   
     void CreateChildren(arma::Col<size_t>& indices,
                         arma::vec& distances,
                         size_t nearSetSize,
                         size_t& farSetSize,
                         size_t& usedSetSize);
   
     void ComputeDistances(const size_t pointIndex,
                           const arma::Col<size_t>& indices,
                           arma::vec& distances,
                           const size_t pointSetSize);
     size_t SplitNearFar(arma::Col<size_t>& indices,
                         arma::vec& distances,
                         const ElemType bound,
                         const size_t pointSetSize);
   
     size_t SortPointSet(arma::Col<size_t>& indices,
                         arma::vec& distances,
                         const size_t childFarSetSize,
                         const size_t childUsedSetSize,
                         const size_t farSetSize);
   
     void MoveToUsedSet(arma::Col<size_t>& indices,
                        arma::vec& distances,
                        size_t& nearSetSize,
                        size_t& farSetSize,
                        size_t& usedSetSize,
                        arma::Col<size_t>& childIndices,
                        const size_t childFarSetSize,
                        const size_t childUsedSetSize);
     size_t PruneFarSet(arma::Col<size_t>& indices,
                        arma::vec& distances,
                        const ElemType bound,
                        const size_t nearSetSize,
                        const size_t pointSetSize);
   
     void RemoveNewImplicitNodes();
   
    protected:
     CoverTree();
   
     friend class cereal::access;
   
    public:
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     size_t DistanceComps() const { return distanceComps; }
     size_t& DistanceComps() { return distanceComps; }
   
    private:
     size_t distanceComps;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "cover_tree_impl.hpp"
   
   // Include the rest of the pieces, if necessary.
   #include "../cover_tree.hpp"
   
   #endif
