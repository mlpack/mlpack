
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_rectangle_tree.hpp:

Program Listing for File rectangle_tree.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_rectangle_tree.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/rectangle_tree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "../hrectbound.hpp"
   #include "../statistic.hpp"
   #include "r_tree_split.hpp"
   #include "r_tree_descent_heuristic.hpp"
   #include "no_auxiliary_information.hpp"
   
   namespace mlpack {
   namespace tree  {
   
   template<typename MetricType = metric::EuclideanDistance,
            typename StatisticType = EmptyStatistic,
            typename MatType = arma::mat,
            typename SplitType = RTreeSplit,
            typename DescentType = RTreeDescentHeuristic,
            template<typename> class AuxiliaryInformationType =
                NoAuxiliaryInformation>
   class RectangleTree
   {
     // The metric *must* be the euclidean distance.
     static_assert(boost::is_same<MetricType, metric::EuclideanDistance>::value,
         "RectangleTree: MetricType must be metric::EuclideanDistance.");
   
    public:
     typedef MatType Mat;
     typedef typename MatType::elem_type ElemType;
     typedef AuxiliaryInformationType<RectangleTree> AuxiliaryInformation;
    private:
     size_t maxNumChildren;
     size_t minNumChildren;
     size_t numChildren;
     std::vector<RectangleTree*> children;
     RectangleTree* parent;
     size_t begin;
     size_t count;
     size_t numDescendants;
     size_t maxLeafSize;
     size_t minLeafSize;
     bound::HRectBound<metric::EuclideanDistance, ElemType> bound;
     StatisticType stat;
     ElemType parentDistance;
     const MatType* dataset;
     bool ownsDataset;
     std::vector<size_t> points;
     AuxiliaryInformationType<RectangleTree> auxiliaryInfo;
   
    public:
     template<typename RuleType>
     class SingleTreeTraverser;
     template<typename RuleType>
     class DualTreeTraverser;
   
     RectangleTree(const MatType& data,
                   const size_t maxLeafSize = 20,
                   const size_t minLeafSize = 8,
                   const size_t maxNumChildren = 5,
                   const size_t minNumChildren = 2,
                   const size_t firstDataIndex = 0);
   
     RectangleTree(MatType&& data,
                   const size_t maxLeafSize = 20,
                   const size_t minLeafSize = 8,
                   const size_t maxNumChildren = 5,
                   const size_t minNumChildren = 2,
                   const size_t firstDataIndex = 0);
   
     explicit RectangleTree(RectangleTree* parentNode,
                            const size_t numMaxChildren = 0);
   
     RectangleTree(const RectangleTree& other,
                   const bool deepCopy = true,
                   RectangleTree* newParent = NULL);
   
     RectangleTree(RectangleTree&& other);
   
     RectangleTree& operator=(const RectangleTree& other);
   
     RectangleTree& operator=(RectangleTree&& other);
   
     template<typename Archive>
     RectangleTree(
         Archive& ar,
         const typename std::enable_if_t<cereal::is_loading<Archive>()>* = 0);
   
     ~RectangleTree();
   
     void SoftDelete();
   
     void NullifyData();
   
     void InsertPoint(const size_t point);
   
     void InsertPoint(const size_t point, std::vector<bool>& relevels);
   
     void InsertNode(RectangleTree* node,
                     const size_t level,
                     std::vector<bool>& relevels);
   
     bool DeletePoint(const size_t point);
   
     bool DeletePoint(const size_t point, std::vector<bool>& relevels);
   
     bool RemoveNode(const RectangleTree* node, std::vector<bool>& relevels);
   
     const RectangleTree* FindByBeginCount(size_t begin, size_t count) const;
   
     RectangleTree* FindByBeginCount(size_t begin, size_t count);
   
     const bound::HRectBound<MetricType>& Bound() const { return bound; }
     bound::HRectBound<MetricType>& Bound() { return bound; }
   
     const StatisticType& Stat() const { return stat; }
     StatisticType& Stat() { return stat; }
   
     const AuxiliaryInformationType<RectangleTree> &AuxiliaryInfo() const
     { return auxiliaryInfo; }
     AuxiliaryInformationType<RectangleTree>& AuxiliaryInfo()
     { return auxiliaryInfo; }
   
     bool IsLeaf() const;
   
     size_t MaxLeafSize() const { return maxLeafSize; }
     size_t& MaxLeafSize() { return maxLeafSize; }
   
     size_t MinLeafSize() const { return minLeafSize; }
     size_t& MinLeafSize() { return minLeafSize; }
   
     size_t MaxNumChildren() const { return maxNumChildren; }
     size_t& MaxNumChildren() { return maxNumChildren; }
   
     size_t MinNumChildren() const { return minNumChildren; }
     size_t& MinNumChildren() { return minNumChildren; }
   
     RectangleTree* Parent() const { return parent; }
     RectangleTree*& Parent() { return parent; }
   
     const MatType& Dataset() const { return *dataset; }
     MatType& Dataset() { return const_cast<MatType&>(*dataset); }
   
     MetricType Metric() const { return MetricType(); }
   
     void Center(arma::vec& center) { bound.Center(center); }
   
     size_t NumChildren() const { return numChildren; }
     size_t& NumChildren() { return numChildren; }
   
     template<typename VecType>
     size_t GetNearestChild(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0);
   
     template<typename VecType>
     size_t GetFurthestChild(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0);
   
     size_t GetNearestChild(const RectangleTree& queryNode);
   
     size_t GetFurthestChild(const RectangleTree& queryNode);
   
     ElemType FurthestPointDistance() const;
   
     ElemType FurthestDescendantDistance() const;
   
     ElemType MinimumBoundDistance() const { return bound.MinWidth() / 2.0; }
   
     ElemType ParentDistance() const { return parentDistance; }
     ElemType& ParentDistance() { return parentDistance; }
   
     inline RectangleTree& Child(const size_t child) const
     {
       return *children[child];
     }
   
     inline RectangleTree& Child(const size_t child)
     {
       return *children[child];
     }
   
     size_t NumPoints() const;
   
     size_t NumDescendants() const;
   
     size_t Descendant(const size_t index) const;
   
     size_t Point(const size_t index) const { return points[index]; }
   
     size_t& Point(const size_t index) { return points[index]; }
   
     ElemType MinDistance(const RectangleTree& other) const
     {
       return bound.MinDistance(other.Bound());
     }
   
     ElemType MaxDistance(const RectangleTree& other) const
     {
       return bound.MaxDistance(other.Bound());
     }
   
     math::RangeType<ElemType> RangeDistance(const RectangleTree& other) const
     {
       return bound.RangeDistance(other.Bound());
     }
   
     template<typename VecType>
     ElemType MinDistance(const VecType& point,
                          typename std::enable_if_t<IsVector<VecType>::value>* = 0)
         const
     {
       return bound.MinDistance(point);
     }
   
     template<typename VecType>
     ElemType MaxDistance(const VecType& point,
                          typename std::enable_if_t<IsVector<VecType>::value>* = 0)
         const
     {
       return bound.MaxDistance(point);
     }
   
     template<typename VecType>
     math::RangeType<ElemType> RangeDistance(
         const VecType& point,
         typename std::enable_if_t<IsVector<VecType>::value>* = 0) const
     {
       return bound.RangeDistance(point);
     }
   
     size_t TreeSize() const;
   
     size_t TreeDepth() const;
   
     size_t Begin() const { return begin; }
     size_t& Begin() { return begin; }
   
     size_t Count() const { return count; }
     size_t& Count() { return count; }
   
    private:
     void SplitNode(std::vector<bool>& relevels);
   
     void BuildStatistics(RectangleTree* node);
   
    protected:
     RectangleTree();
   
     friend class cereal::access;
   
     friend DescentType;
   
     friend SplitType;
   
     friend AuxiliaryInformation;
   
    public:
     void CondenseTree(const arma::vec& point,
                       std::vector<bool>& relevels,
                       const bool usePoint);
   
     bool ShrinkBoundForPoint(const arma::vec& point);
   
     bool ShrinkBoundForBound(const bound::HRectBound<MetricType>& changedBound);
   
     RectangleTree* ExactClone();
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "rectangle_tree_impl.hpp"
   
   #endif
