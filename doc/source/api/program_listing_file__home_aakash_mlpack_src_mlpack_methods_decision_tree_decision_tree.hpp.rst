
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_decision_tree_decision_tree.hpp:

Program Listing for File decision_tree.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_decision_tree_decision_tree.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/decision_tree/decision_tree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DECISION_TREE_DECISION_TREE_HPP
   #define MLPACK_METHODS_DECISION_TREE_DECISION_TREE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "gini_gain.hpp"
   #include "information_gain.hpp"
   #include "best_binary_numeric_split.hpp"
   #include "random_binary_numeric_split.hpp"
   #include "all_categorical_split.hpp"
   #include "all_dimension_select.hpp"
   #include <type_traits>
   
   namespace mlpack {
   namespace tree {
   
   template<typename FitnessFunction = GiniGain,
            template<typename> class NumericSplitType = BestBinaryNumericSplit,
            template<typename> class CategoricalSplitType = AllCategoricalSplit,
            typename DimensionSelectionType = AllDimensionSelect,
            bool NoRecursion = false>
   class DecisionTree :
       public NumericSplitType<FitnessFunction>::AuxiliarySplitInfo,
       public CategoricalSplitType<FitnessFunction>::AuxiliarySplitInfo
   {
    public:
     typedef NumericSplitType<FitnessFunction> NumericSplit;
     typedef CategoricalSplitType<FitnessFunction> CategoricalSplit;
     typedef DimensionSelectionType DimensionSelection;
   
     template<typename MatType, typename LabelsType>
     DecisionTree(MatType data,
                  const data::DatasetInfo& datasetInfo,
                  LabelsType labels,
                  const size_t numClasses,
                  const size_t minimumLeafSize = 10,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename MatType, typename LabelsType>
     DecisionTree(MatType data,
                  LabelsType labels,
                  const size_t numClasses,
                  const size_t minimumLeafSize = 10,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename MatType, typename LabelsType, typename WeightsType>
     DecisionTree(
         MatType data,
         const data::DatasetInfo& datasetInfo,
         LabelsType labels,
         const size_t numClasses,
         WeightsType weights,
         const size_t minimumLeafSize = 10,
         const double minimumGainSplit = 1e-7,
         const size_t maximumDepth = 0,
         DimensionSelectionType dimensionSelector = DimensionSelectionType(),
         const std::enable_if_t<arma::is_arma_type<
             typename std::remove_reference<WeightsType>::type>::value>* = 0);
   
     template<typename MatType, typename LabelsType, typename WeightsType>
     DecisionTree(
         const DecisionTree& other,
         MatType data,
         const data::DatasetInfo& datasetInfo,
         LabelsType labels,
         const size_t numClasses,
         WeightsType weights,
         const size_t minimumLeafSize = 10,
         const double minimumGainSplit = 1e-7,
         const std::enable_if_t<arma::is_arma_type<
             typename std::remove_reference<WeightsType>::type>::value>* = 0);
     template<typename MatType, typename LabelsType, typename WeightsType>
     DecisionTree(
         MatType data,
         LabelsType labels,
         const size_t numClasses,
         WeightsType weights,
         const size_t minimumLeafSize = 10,
         const double minimumGainSplit = 1e-7,
         const size_t maximumDepth = 0,
         DimensionSelectionType dimensionSelector = DimensionSelectionType(),
         const std::enable_if_t<arma::is_arma_type<
             typename std::remove_reference<WeightsType>::type>::value>* = 0);
   
     template<typename MatType, typename LabelsType, typename WeightsType>
     DecisionTree(
         const DecisionTree& other,
         MatType data,
         LabelsType labels,
         const size_t numClasses,
         WeightsType weights,
         const size_t minimumLeafSize = 10,
         const double minimumGainSplit = 1e-7,
         const size_t maximumDepth = 0,
         DimensionSelectionType dimensionSelector = DimensionSelectionType(),
         const std::enable_if_t<arma::is_arma_type<
             typename std::remove_reference<WeightsType>::type>::value>* = 0);
   
     DecisionTree(const size_t numClasses = 1);
   
     DecisionTree(const DecisionTree& other);
   
     DecisionTree(DecisionTree&& other);
   
     DecisionTree& operator=(const DecisionTree& other);
   
     DecisionTree& operator=(DecisionTree&& other);
   
     ~DecisionTree();
   
     template<typename MatType, typename LabelsType>
     double Train(MatType data,
                  const data::DatasetInfo& datasetInfo,
                  LabelsType labels,
                  const size_t numClasses,
                  const size_t minimumLeafSize = 10,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename MatType, typename LabelsType>
     double Train(MatType data,
                  LabelsType labels,
                  const size_t numClasses,
                  const size_t minimumLeafSize = 10,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename MatType, typename LabelsType, typename WeightsType>
     double Train(MatType data,
                  const data::DatasetInfo& datasetInfo,
                  LabelsType labels,
                  const size_t numClasses,
                  WeightsType weights,
                  const size_t minimumLeafSize = 10,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType(),
                  const std::enable_if_t<arma::is_arma_type<typename
                      std::remove_reference<WeightsType>::type>::value>* = 0);
   
     template<typename MatType, typename LabelsType, typename WeightsType>
     double Train(MatType data,
                  LabelsType labels,
                  const size_t numClasses,
                  WeightsType weights,
                  const size_t minimumLeafSize = 10,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType(),
                  const std::enable_if_t<arma::is_arma_type<typename
                      std::remove_reference<WeightsType>::type>::value>* = 0);
   
     template<typename VecType>
     size_t Classify(const VecType& point) const;
   
     template<typename VecType>
     void Classify(const VecType& point,
                   size_t& prediction,
                   arma::vec& probabilities) const;
   
     template<typename MatType>
     void Classify(const MatType& data,
                   arma::Row<size_t>& predictions) const;
   
     template<typename MatType>
     void Classify(const MatType& data,
                   arma::Row<size_t>& predictions,
                   arma::mat& probabilities) const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     size_t NumChildren() const { return children.size(); }
   
     const DecisionTree& Child(const size_t i) const { return *children[i]; }
     DecisionTree& Child(const size_t i) { return *children[i]; }
   
     size_t SplitDimension() const { return splitDimension; }
   
     template<typename VecType>
     size_t CalculateDirection(const VecType& point) const;
   
     size_t NumClasses() const;
   
    private:
     std::vector<DecisionTree*> children;
     size_t splitDimension;
     size_t dimensionTypeOrMajorityClass;
     arma::vec classProbabilities;
   
     typedef typename NumericSplit::AuxiliarySplitInfo
         NumericAuxiliarySplitInfo;
     typedef typename CategoricalSplit::AuxiliarySplitInfo
         CategoricalAuxiliarySplitInfo;
   
     template<bool UseWeights, typename RowType, typename WeightsRowType>
     void CalculateClassProbabilities(const RowType& labels,
                                      const size_t numClasses,
                                      const WeightsRowType& weights);
   
     template<bool UseWeights, typename MatType>
     double Train(MatType& data,
                  const size_t begin,
                  const size_t count,
                  const data::DatasetInfo& datasetInfo,
                  arma::Row<size_t>& labels,
                  const size_t numClasses,
                  arma::rowvec& weights,
                  const size_t minimumLeafSize,
                  const double minimumGainSplit,
                  const size_t maximumDepth,
                  DimensionSelectionType& dimensionSelector);
   
     template<bool UseWeights, typename MatType>
     double Train(MatType& data,
                  const size_t begin,
                  const size_t count,
                  arma::Row<size_t>& labels,
                  const size_t numClasses,
                  arma::rowvec& weights,
                  const size_t minimumLeafSize,
                  const double minimumGainSplit,
                  const size_t maximumDepth,
                  DimensionSelectionType& dimensionSelector);
   };
   
   template<typename FitnessFunction = GiniGain,
            template<typename> class NumericSplitType = BestBinaryNumericSplit,
            template<typename> class CategoricalSplitType = AllCategoricalSplit,
            typename DimensionSelectType = AllDimensionSelect>
   using DecisionStump = DecisionTree<FitnessFunction,
                                      NumericSplitType,
                                      CategoricalSplitType,
                                      DimensionSelectType,
                                      false>;
   
   typedef DecisionTree<InformationGain,
                        BestBinaryNumericSplit,
                        AllCategoricalSplit,
                        AllDimensionSelect,
                        true> ID3DecisionStump;
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "decision_tree_impl.hpp"
   
   #endif
