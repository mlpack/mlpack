
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_octree_octree.hpp:

Program Listing for File octree.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_octree_octree.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/octree/octree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_OCTREE_OCTREE_HPP
   #define MLPACK_CORE_TREE_OCTREE_OCTREE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "../hrectbound.hpp"
   #include "../statistic.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType = metric::EuclideanDistance,
            typename StatisticType = EmptyStatistic,
            typename MatType = arma::mat>
   class Octree
   {
    public:
     typedef MatType Mat;
     typedef typename MatType::elem_type ElemType;
   
     template<typename RuleType>
     class SingleTreeTraverser;
   
     template<typename RuleType>
     class DualTreeTraverser;
   
    private:
     std::vector<Octree*> children;
   
     size_t begin;
     size_t count;
     bound::HRectBound<MetricType> bound;
     MatType* dataset;
     Octree* parent;
     StatisticType stat;
     ElemType parentDistance;
     ElemType furthestDescendantDistance;
     MetricType metric;
   
    public:
     Octree(const MatType& data, const size_t maxLeafSize = 20);
   
     Octree(const MatType& data,
            std::vector<size_t>& oldFromNew,
            const size_t maxLeafSize = 20);
   
     Octree(const MatType& data,
            std::vector<size_t>& oldFromNew,
            std::vector<size_t>& newFromOld,
            const size_t maxLeafSize = 20);
   
     Octree(MatType&& data, const size_t maxLeafSize = 20);
   
     Octree(MatType&& data,
            std::vector<size_t>& oldFromNew,
            const size_t maxLeafSize = 20);
   
     Octree(MatType&& data,
            std::vector<size_t>& oldFromNew,
            std::vector<size_t>& newFromOld,
            const size_t maxLeafSize = 20);
   
     Octree(Octree* parent,
            const size_t begin,
            const size_t count,
            const arma::vec& center,
            const double width,
            const size_t maxLeafSize = 20);
   
     Octree(Octree* parent,
            const size_t begin,
            const size_t count,
            std::vector<size_t>& oldFromNew,
            const arma::vec& center,
            const double width,
            const size_t maxLeafSize = 20);
   
     Octree(const Octree& other);
   
     Octree(Octree&& other);
   
     Octree& operator=(const Octree& other);
   
     Octree& operator=(Octree&& other);
   
     template<typename Archive>
     Octree(
         Archive& ar,
         const typename std::enable_if_t<cereal::is_loading<Archive>()>* = 0);
   
     ~Octree();
   
     const MatType& Dataset() const { return *dataset; }
   
     Octree* Parent() const { return parent; }
     Octree*& Parent() { return parent; }
   
     const bound::HRectBound<MetricType>& Bound() const { return bound; }
     bound::HRectBound<MetricType>& Bound() { return bound; }
   
     const StatisticType& Stat() const { return stat; }
     StatisticType& Stat() { return stat; }
   
     size_t NumChildren() const;
   
     MetricType Metric() const { return MetricType(); }
   
     template<typename VecType>
     size_t GetNearestChild(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
   
     template<typename VecType>
     size_t GetFurthestChild(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
   
     bool IsLeaf() const { return NumChildren() == 0; }
   
     size_t GetNearestChild(const Octree& queryNode) const;
   
     size_t GetFurthestChild(const Octree& queryNode) const;
   
     ElemType FurthestPointDistance() const;
   
     ElemType FurthestDescendantDistance() const;
   
     ElemType MinimumBoundDistance() const;
   
     ElemType ParentDistance() const { return parentDistance; }
     ElemType& ParentDistance() { return parentDistance; }
   
     const Octree& Child(const size_t child) const { return *children[child]; }
   
     Octree& Child(const size_t child) { return *children[child]; }
   
     Octree*& ChildPtr(const size_t child) { return children[child]; }
   
     size_t NumPoints() const;
   
     size_t NumDescendants() const;
   
     size_t Descendant(const size_t index) const;
   
     size_t Point(const size_t index) const;
   
     ElemType MinDistance(const Octree& other) const;
     ElemType MaxDistance(const Octree& other) const;
     math::RangeType<ElemType> RangeDistance(const Octree& other) const;
   
     template<typename VecType>
     ElemType MinDistance(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
     template<typename VecType>
     ElemType MaxDistance(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
     template<typename VecType>
     math::RangeType<ElemType> RangeDistance(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;
   
     void Center(arma::vec& center) const { bound.Center(center); }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    protected:
     Octree();
   
     friend class cereal::access;
   
    private:
     void SplitNode(const arma::vec& center,
                    const double width,
                    const size_t maxLeafSize);
   
     void SplitNode(const arma::vec& center,
                    const double width,
                    std::vector<size_t>& oldFromNew,
                    const size_t maxLeafSize);
   
     struct SplitType
     {
       struct SplitInfo
       {
         SplitInfo(const size_t d, const arma::vec& c) : d(d), center(c) {}
   
         size_t d;
         const arma::vec& center;
       };
   
       template<typename VecType>
       static bool AssignToLeftNode(const VecType& point, const SplitInfo& s)
       {
         return point[s.d] < s.center[s.d];
       }
     };
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "octree_impl.hpp"
   
   #endif
