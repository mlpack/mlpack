
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_hoeffding_tree_model.hpp:

Program Listing for File hoeffding_tree_model.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_hoeffding_tree_model.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hoeffding_trees/hoeffding_tree_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_TREE_MODEL_HPP
   #define MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_TREE_MODEL_HPP
   
   #include "hoeffding_tree.hpp"
   #include "binary_numeric_split.hpp"
   #include "information_gain.hpp"
   
   namespace mlpack {
   namespace tree {
   
   class HoeffdingTreeModel
   {
    public:
     enum TreeType
     {
       GINI_HOEFFDING,
       GINI_BINARY,
       INFO_HOEFFDING,
       INFO_BINARY
     };
   
     typedef HoeffdingTree<GiniImpurity, HoeffdingDoubleNumericSplit,
         HoeffdingCategoricalSplit> GiniHoeffdingTreeType;
     typedef HoeffdingTree<GiniImpurity, BinaryDoubleNumericSplit,
         HoeffdingCategoricalSplit> GiniBinaryTreeType;
     typedef HoeffdingTree<HoeffdingInformationGain, HoeffdingDoubleNumericSplit,
         HoeffdingCategoricalSplit> InfoHoeffdingTreeType;
     typedef HoeffdingTree<HoeffdingInformationGain, BinaryDoubleNumericSplit,
         HoeffdingCategoricalSplit> InfoBinaryTreeType;
   
     HoeffdingTreeModel(const TreeType& type = GINI_HOEFFDING);
   
     HoeffdingTreeModel(const HoeffdingTreeModel& other);
   
     HoeffdingTreeModel(HoeffdingTreeModel&& other);
   
     HoeffdingTreeModel& operator=(const HoeffdingTreeModel& other);
   
     HoeffdingTreeModel& operator=(HoeffdingTreeModel&& other);
   
     ~HoeffdingTreeModel();
   
     void BuildModel(const arma::mat& dataset,
                     const data::DatasetInfo& datasetInfo,
                     const arma::Row<size_t>& labels,
                     const size_t numClasses,
                     const bool batchTraining,
                     const double successProbability,
                     const size_t maxSamples,
                     const size_t checkInterval,
                     const size_t minSamples,
                     const size_t bins,
                     const size_t observationsBeforeBinning);
   
     void Train(const arma::mat& dataset,
                const arma::Row<size_t>& labels,
                const bool batchTraining);
   
     void Classify(const arma::mat& dataset,
                   arma::Row<size_t>& predictions) const;
   
     void Classify(const arma::mat& dataset,
                   arma::Row<size_t>& predictions,
                   arma::rowvec& probabilities) const;
   
     size_t NumNodes() const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       // Clear memory if needed.
       if (cereal::is_loading<Archive>())
       {
         delete giniHoeffdingTree;
         delete giniBinaryTree;
         delete infoHoeffdingTree;
         delete infoBinaryTree;
   
         giniHoeffdingTree = NULL;
         giniBinaryTree = NULL;
         infoHoeffdingTree = NULL;
         infoBinaryTree = NULL;
       }
   
       ar(CEREAL_NVP(type));
   
       // Fake dataset info may be needed to create fake trees.
       data::DatasetInfo info;
       if (type == GINI_HOEFFDING)
         ar(CEREAL_POINTER(giniHoeffdingTree));
       else if (type == GINI_BINARY)
         ar(CEREAL_POINTER(giniBinaryTree));
       else if (type == INFO_HOEFFDING)
         ar(CEREAL_POINTER(infoHoeffdingTree));
       else if (type == INFO_BINARY)
         ar(CEREAL_POINTER(infoBinaryTree));
     }
   
    private:
     TreeType type;
   
     GiniHoeffdingTreeType* giniHoeffdingTree;
   
     GiniBinaryTreeType* giniBinaryTree;
   
     InfoHoeffdingTreeType* infoHoeffdingTree;
   
     InfoBinaryTreeType* infoBinaryTree;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
