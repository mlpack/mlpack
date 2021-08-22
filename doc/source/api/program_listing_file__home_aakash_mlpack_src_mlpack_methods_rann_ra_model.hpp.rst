
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_rann_ra_model.hpp:

Program Listing for File ra_model.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_rann_ra_model.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/rann/ra_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RANN_RA_MODEL_HPP
   #define MLPACK_METHODS_RANN_RA_MODEL_HPP
   
   #include <mlpack/core/tree/binary_space_tree.hpp>
   #include <mlpack/core/tree/cover_tree.hpp>
   #include <mlpack/core/tree/rectangle_tree.hpp>
   #include <mlpack/core/tree/octree.hpp>
   #include "ra_search.hpp"
   
   namespace mlpack {
   namespace neighbor {
   
   class RAWrapperBase
   {
    public:
     RAWrapperBase() { }
   
     virtual RAWrapperBase* Clone() const = 0;
   
     virtual ~RAWrapperBase() { }
   
     virtual const arma::mat& Dataset() const = 0;
   
     virtual size_t SingleSampleLimit() const = 0;
     virtual size_t& SingleSampleLimit() = 0;
   
     virtual bool FirstLeafExact() const = 0;
     virtual bool& FirstLeafExact() = 0;
   
     virtual bool SampleAtLeaves() const = 0;
     virtual bool& SampleAtLeaves() = 0;
   
     virtual double Alpha() const = 0;
     virtual double& Alpha() = 0;
   
     virtual double Tau() const = 0;
     virtual double& Tau() = 0;
   
     virtual bool SingleMode() const = 0;
     virtual bool& SingleMode() = 0;
   
     virtual bool Naive() const = 0;
     virtual bool& Naive() = 0;
   
     virtual void Train(util::Timers& timers,
                        arma::mat&& referenceSet,
                        const size_t leafSize) = 0;
   
     virtual void Search(util::Timers& timers,
                         arma::mat&& querySet,
                         const size_t k,
                         arma::Mat<size_t>& neighbors,
                         arma::mat& distances,
                         const size_t leafSize) = 0;
   
     virtual void Search(util::Timers& timers,
                         const size_t k,
                         arma::Mat<size_t>& neighbors,
                         arma::mat& distances) = 0;
   };
   
   template<template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   class RAWrapper : public RAWrapperBase
   {
    public:
     RAWrapper(const bool singleMode, const bool naive) :
         ra(singleMode, naive)
     {
       // Nothing else to do.
     }
   
     virtual ~RAWrapper() { }
   
     virtual RAWrapper* Clone() const { return new RAWrapper(*this); }
   
     const arma::mat& Dataset() const { return ra.ReferenceSet(); }
   
     size_t SingleSampleLimit() const { return ra.SingleSampleLimit(); }
     size_t& SingleSampleLimit() { return ra.SingleSampleLimit(); }
   
     bool FirstLeafExact() const { return ra.FirstLeafExact(); }
     bool& FirstLeafExact() { return ra.FirstLeafExact(); }
   
     bool SampleAtLeaves() const { return ra.SampleAtLeaves(); }
     bool& SampleAtLeaves() { return ra.SampleAtLeaves(); }
   
     double Alpha() const { return ra.Alpha(); }
     double& Alpha() { return ra.Alpha(); }
   
     double Tau() const { return ra.Tau(); }
     double& Tau() { return ra.Tau(); }
   
     bool SingleMode() const { return ra.SingleMode(); }
     bool& SingleMode() { return ra.SingleMode(); }
   
     bool Naive() const { return ra.Naive(); }
     bool& Naive() { return ra.Naive(); }
   
     virtual void Train(util::Timers& timers,
                        arma::mat&& referenceSet,
                        const size_t /* leafSize */);
   
     virtual void Search(util::Timers& timers,
                         arma::mat&& querySet,
                         const size_t k,
                         arma::Mat<size_t>& neighbors,
                         arma::mat& distances,
                         const size_t /* leafSize */);
   
     virtual void Search(util::Timers& timers,
                         const size_t k,
                         arma::Mat<size_t>& neighbors,
                         arma::mat& distances);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(ra));
     }
   
    protected:
     typedef RASearch<NearestNeighborSort,
                      metric::EuclideanDistance,
                      arma::mat,
                      TreeType> RAType;
   
     RAType ra;
   };
   
   template<template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   class LeafSizeRAWrapper : public RAWrapper<TreeType>
   {
    public:
     LeafSizeRAWrapper(const bool singleMode, const bool naive) :
         RAWrapper<TreeType>(singleMode, naive)
     {
       // Nothing else to do.
     }
   
     virtual ~LeafSizeRAWrapper() { }
   
     virtual LeafSizeRAWrapper* Clone() const
     {
       return new LeafSizeRAWrapper(*this);
     }
   
     virtual void Train(util::Timers& timers,
                        arma::mat&& referenceSet,
                        const size_t leafSize);
   
     virtual void Search(util::Timers& timers,
                         arma::mat&& querySet,
                         const size_t k,
                         arma::Mat<size_t>& neighbors,
                         arma::mat& distances,
                         const size_t leafSize);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(ra));
     }
   
    protected:
     using RAWrapper<TreeType>::ra;
   };
   
   class RAModel
   {
    public:
     enum TreeTypes
     {
       KD_TREE,
       COVER_TREE,
       R_TREE,
       R_STAR_TREE,
       X_TREE,
       HILBERT_R_TREE,
       R_PLUS_TREE,
       R_PLUS_PLUS_TREE,
       UB_TREE,
       OCTREE
     };
   
    private:
     TreeTypes treeType;
     size_t leafSize;
   
     bool randomBasis;
     arma::mat q;
   
     RAWrapperBase* raSearch;
   
    public:
     RAModel(TreeTypes treeType = TreeTypes::KD_TREE, bool randomBasis = false);
   
     RAModel(const RAModel& other);
   
     RAModel(RAModel&& other);
   
     RAModel& operator=(const RAModel& other);
   
     RAModel& operator=(RAModel&& other);
   
     ~RAModel();
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     const arma::mat& Dataset() const { return raSearch->Dataset(); }
   
     bool SingleMode() const { return raSearch->SingleMode(); }
     bool& SingleMode() { return raSearch->SingleMode(); }
   
     bool Naive() const { return raSearch->Naive(); }
     bool& Naive() { return raSearch->Naive(); }
   
     double Tau() const { return raSearch->Tau(); }
     double& Tau() { return raSearch->Tau(); }
   
     double Alpha() const { return raSearch->Alpha(); }
     double& Alpha() { return raSearch->Alpha(); }
   
     bool SampleAtLeaves() const { return raSearch->SampleAtLeaves(); }
     bool& SampleAtLeaves() { return raSearch->SampleAtLeaves(); }
   
     bool FirstLeafExact() const { return raSearch->FirstLeafExact(); }
     bool& FirstLeafExact() { return raSearch->FirstLeafExact(); }
   
     size_t SingleSampleLimit() const { return raSearch->SingleSampleLimit(); }
     size_t& SingleSampleLimit() { return raSearch->SingleSampleLimit(); }
   
     size_t LeafSize() const { return leafSize; }
     size_t& LeafSize() { return leafSize; }
   
     TreeTypes TreeType() const { return treeType; }
     TreeTypes& TreeType() { return treeType; }
   
     bool RandomBasis() const { return randomBasis; }
     bool& RandomBasis() { return randomBasis; }
   
     void InitializeModel(const bool naive, const bool singleMode);
   
     void BuildModel(util::Timers& timers,
                     arma::mat&& referenceSet,
                     const size_t leafSize,
                     const bool naive,
                     const bool singleMode);
   
     void Search(util::Timers& timers,
                 arma::mat&& querySet,
                 const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances);
   
     void Search(util::Timers& timers,
                 const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances);
   
     std::string TreeName() const;
   };
   
   } // namespace neighbor
   } // namespace mlpack
   
   #include "ra_model_impl.hpp"
   
   #endif
