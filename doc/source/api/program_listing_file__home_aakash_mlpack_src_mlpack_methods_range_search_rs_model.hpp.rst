
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_range_search_rs_model.hpp:

Program Listing for File rs_model.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_range_search_rs_model.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/range_search/rs_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_HPP
   #define MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_HPP
   
   #include <mlpack/core/tree/binary_space_tree.hpp>
   #include <mlpack/core/tree/cover_tree.hpp>
   #include <mlpack/core/tree/rectangle_tree.hpp>
   #include <mlpack/core/tree/octree.hpp>
   
   #include "range_search.hpp"
   
   namespace mlpack {
   namespace range {
   
   class RSWrapperBase
   {
    public:
     RSWrapperBase() { }
   
     virtual RSWrapperBase* Clone() const = 0;
   
     virtual ~RSWrapperBase() { }
   
     virtual const arma::mat& Dataset() const = 0;
   
     virtual bool SingleMode() const = 0;
     virtual bool& SingleMode() = 0;
   
     virtual bool Naive() const = 0;
     virtual bool& Naive() = 0;
   
     virtual void Train(arma::mat&& referenceSet,
                        const size_t leafSize) = 0;
   
     virtual void Search(arma::mat&& querySet,
                         const math::Range& range,
                         std::vector<std::vector<size_t>>& neighbors,
                         std::vector<std::vector<double>>& distances,
                         const size_t leafSize) = 0;
   
     virtual void Search(const math::Range& range,
                         std::vector<std::vector<size_t>>& neighbors,
                         std::vector<std::vector<double>>& distances) = 0;
   };
   
   template<template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   class RSWrapper : public RSWrapperBase
   {
    public:
     RSWrapper(const bool singleMode, const bool naive) :
         rs(singleMode, naive)
     {
       // Nothing else to do.
     }
   
     virtual RSWrapper* Clone() const { return new RSWrapper(*this); }
   
     virtual ~RSWrapper() { }
   
     const arma::mat& Dataset() const { return rs.ReferenceSet(); }
   
     bool SingleMode() const { return rs.SingleMode(); }
     bool& SingleMode() { return rs.SingleMode(); }
   
     bool Naive() const { return rs.Naive(); }
     bool& Naive() { return rs.Naive(); }
   
     virtual void Train(arma::mat&& referenceSet,
                        const size_t /* leafSize */);
   
     virtual void Search(arma::mat&& querySet,
                         const math::Range& range,
                         std::vector<std::vector<size_t>>& neighbors,
                         std::vector<std::vector<double>>& distances,
                         const size_t /* leafSize */);
   
     virtual void Search(const math::Range& range,
                         std::vector<std::vector<size_t>>& neighbors,
                         std::vector<std::vector<double>>& distances);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(rs));
     }
   
    protected:
     typedef RangeSearch<metric::EuclideanDistance, arma::mat, TreeType> RSType;
   
     RSType rs;
   };
   
   template<template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   class LeafSizeRSWrapper : public RSWrapper<TreeType>
   {
    public:
     LeafSizeRSWrapper(const bool singleMode, const bool naive) :
         RSWrapper<TreeType>(singleMode, naive)
     {
       // Nothing else to do.
     }
   
     virtual ~LeafSizeRSWrapper() { }
   
     virtual LeafSizeRSWrapper* Clone() const
     {
       return new LeafSizeRSWrapper(*this);
     }
   
     virtual void Train(arma::mat&& referenceSet,
                        const size_t leafSize);
   
     virtual void Search(arma::mat&& querySet,
                         const math::Range& range,
                         std::vector<std::vector<size_t>>& neighbors,
                         std::vector<std::vector<double>>& distances,
                         const size_t leafSize);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(rs));
     }
   
    protected:
     using RSWrapper<TreeType>::rs;
   };
   
   class RSModel
   {
    public:
     enum TreeTypes
     {
       KD_TREE,
       COVER_TREE,
       R_TREE,
       R_STAR_TREE,
       BALL_TREE,
       X_TREE,
       HILBERT_R_TREE,
       R_PLUS_TREE,
       R_PLUS_PLUS_TREE,
       VP_TREE,
       RP_TREE,
       MAX_RP_TREE,
       UB_TREE,
       OCTREE
     };
   
     RSModel(const TreeTypes treeType = TreeTypes::KD_TREE,
             const bool randomBasis = false);
   
     RSModel(const RSModel& other);
   
     RSModel(RSModel&& other);
   
     RSModel& operator=(const RSModel& other);
   
     RSModel& operator=(RSModel&& other);
   
     ~RSModel();
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     const arma::mat& Dataset() const { return rSearch->Dataset(); }
   
     bool SingleMode() const { return rSearch->SingleMode(); }
     bool& SingleMode() { return rSearch->SingleMode(); }
   
     bool Naive() const { return rSearch->Naive(); }
     bool& Naive() { return rSearch->Naive(); }
   
     size_t LeafSize() const { return leafSize; }
     size_t& LeafSize() { return leafSize; }
   
     TreeTypes TreeType() const { return treeType; }
     TreeTypes& TreeType() { return treeType; }
   
     bool RandomBasis() const { return randomBasis; }
     bool& RandomBasis() { return randomBasis; }
   
     void InitializeModel(const bool naive, const bool singleMode);
   
     void BuildModel(arma::mat&& referenceSet,
                     const size_t leafSize,
                     const bool naive,
                     const bool singleMode);
   
     void Search(arma::mat&& querySet,
                 const math::Range& range,
                 std::vector<std::vector<size_t>>& neighbors,
                 std::vector<std::vector<double>>& distances);
   
     void Search(const math::Range& range,
                 std::vector<std::vector<size_t>>& neighbors,
                 std::vector<std::vector<double>>& distances);
   
    private:
     TreeTypes treeType;
     size_t leafSize;
   
     bool randomBasis;
     arma::mat q;
   
     RSWrapperBase* rSearch;
   
     std::string TreeName() const;
   
     void CleanMemory();
   };
   
   } // namespace range
   } // namespace mlpack
   
   // Include implementation (of serialize() and templated wrapper classes).
   #include "rs_model_impl.hpp"
   
   #endif
