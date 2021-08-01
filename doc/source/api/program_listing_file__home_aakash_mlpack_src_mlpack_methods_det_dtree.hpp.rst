
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_det_dtree.hpp:

Program Listing for File dtree.hpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_det_dtree.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/det/dtree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DET_DTREE_HPP
   #define MLPACK_METHODS_DET_DTREE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace det  {
   
   template<typename MatType = arma::mat,
            typename TagType = int>
   class DTree
   {
    public:
     typedef typename MatType::elem_type  ElemType;
     typedef typename MatType::vec_type   VecType;
     typedef typename arma::Col<ElemType> StatType;
   
     DTree();
   
     DTree(const DTree& obj);
   
     DTree& operator=(const DTree& obj);
   
     DTree(DTree&& obj);
   
     DTree& operator=(DTree&& obj);
   
     DTree(const StatType& maxVals,
           const StatType& minVals,
           const size_t totalPoints);
   
     DTree(MatType& data);
   
     DTree(const StatType& maxVals,
           const StatType& minVals,
           const size_t start,
           const size_t end,
           const double logNegError);
   
     DTree(const StatType& maxVals,
           const StatType& minVals,
           const size_t totalPoints,
           const size_t start,
           const size_t end);
   
     ~DTree();
   
     double Grow(MatType& data,
                 arma::Col<size_t>& oldFromNew,
                 const bool useVolReg = false,
                 const size_t maxLeafSize = 10,
                 const size_t minLeafSize = 5);
   
     double PruneAndUpdate(const double oldAlpha,
                           const size_t points,
                           const bool useVolReg = false);
   
     double ComputeValue(const VecType& query) const;
   
     TagType TagTree(const TagType& tag = 0, bool everyNode = false);
   
   
     TagType FindBucket(const VecType& query) const;
   
   
     void ComputeVariableImportance(arma::vec& importances) const;
   
     double LogNegativeError(const size_t totalPoints) const;
   
     bool WithinRange(const VecType& query) const;
   
    private:
     // The indices in the complete set of points
     // (after all forms of swapping in the original data
     // matrix to align all the points in a node
     // consecutively in the matrix. The 'old_from_new' array
     // maps the points back to their original indices.
   
     size_t start;
     size_t end;
   
     StatType maxVals;
     StatType minVals;
   
     size_t splitDim;
   
     ElemType splitValue;
   
     double logNegError;
   
     double subtreeLeavesLogNegError;
   
     size_t subtreeLeaves;
   
     bool root;
   
     double ratio;
   
     double logVolume;
   
     TagType bucketTag;
   
     double alphaUpper;
   
     DTree* left;
     DTree* right;
   
    public:
     size_t Start() const { return start; }
     size_t End() const { return end; }
     size_t SplitDim() const { return splitDim; }
     ElemType SplitValue() const { return splitValue; }
     double LogNegError() const { return logNegError; }
     double SubtreeLeavesLogNegError() const { return subtreeLeavesLogNegError; }
     size_t SubtreeLeaves() const { return subtreeLeaves; }
     double Ratio() const { return ratio; }
     double LogVolume() const { return logVolume; }
     DTree* Left() const { return left; }
     DTree* Right() const { return right; }
     bool Root() const { return root; }
     double AlphaUpper() const { return alphaUpper; }
     TagType BucketTag() const { return bucketTag; }
     size_t NumChildren() const { return !left ? 0 : 2; }
   
     DTree& Child(const size_t child) const { return !child ? *left : *right; }
   
     DTree*& ChildPtr(const size_t child) { return (!child) ? left : right; }
   
     const StatType& MaxVals() const { return maxVals; }
   
     const StatType& MinVals() const { return minVals; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     // Utility methods.
   
     bool FindSplit(const MatType& data,
                    size_t& splitDim,
                    ElemType& splitValue,
                    double& leftError,
                    double& rightError,
                    const size_t minLeafSize = 5) const;
   
     size_t SplitData(MatType& data,
                      const size_t splitDim,
                      const ElemType splitValue,
                      arma::Col<size_t>& oldFromNew) const;
   
     void  FillMinMax(const StatType& mins,
                      const StatType& maxs);
   };
   
   } // namespace det
   } // namespace mlpack
   
   #include "dtree_impl.hpp"
   
   #endif // MLPACK_METHODS_DET_DTREE_HPP
