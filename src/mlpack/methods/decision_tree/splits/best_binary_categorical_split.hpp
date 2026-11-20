/**
 * @file methods/decision_tree/splits/best_binary_categorical_split.hpp
 * @author Nikolay Apanasov (nikolay@apanasov.org)
 *
 * A tree splitter that finds the best binary categorical split.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_SPLITS_BEST_BINARY_CATEGORICAL_SPLIT_HPP
#define MLPACK_METHODS_DECISION_TREE_SPLITS_BEST_BINARY_CATEGORICAL_SPLIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The BestBinaryCategoricalSplit is a splitting function for decision trees
 * that will exhaustively search a categorical dimension for the best binary
 * split of a variable vₖ. This is a generic splitting strategy and can be
 * used for both regression and classification.
 *
 * In the case of binary outcomes, it shown in CART[4.2] by Breiman et al.
 * that if we order the categories by the proportion that fall in class C₁,
 * and then split vₖ as if it was a numeric type, then the result is optimal.
 * Surprising, but true. In the case of multiple classes, there is no such
 * simplification. This method will search through all the 2ʲ possible
 * partitions (Gₗ, Gᵣ) of the categories C₀, ..., Cⱼ₋₁, every time assigning
 * samples with vₖ ∈ Gₗ to left tree Tₗ and those with vₖ ∈ Gᵣ to right
 * tree Tᵣ.
 *
 * Warning: in the classification setting with multiple outcomes, this
 * algorithm is exponential in the number of categories. Therefore
 * BestBinaryCategoricalSplit should not be chosen when there are multiple
 * classes and many categories.
 *
 * @book{CART,
 *   author = {Breiman, L. and Friedman, J. and Olshen, R. and Stone, C.},
 *   year = {1984},
 *   title = {Classification and Regression Trees},
 *   publisher = {Chapman \& Hall}
 * }
 *
 * In the regression setting, the algorithm is similar to the preceding linear-
 * time split for the case of binary outcomes. The correctness of the algorithm
 * for a quantitative response under l₂ loss is due to Fisher.
 *
 * @article{Fisher58,
 *   author = {Fisher, W.},
 *   year = {1958},
 *   title = {On Grouping for Maximum Homogeniety},
 *   journal = {Journal of the American Statistical Association},
 *   volume = {53},
 *   pages = {789–798}
 * }
 *
 * @tparam FitnessFunction Fitness function to use to calculate gain.
 */
template<typename FitnessFunction>
class BestBinaryCategoricalSplit
{
 public:
  // No extra info needed for split.
  class AuxiliarySplitInfo { };
  // Allow access to the numeric split type.
  using NumericSplit = BestBinaryNumericSplit<FitnessFunction>;
  // For calls to the numeric splitter.
  using NumericAux =
      typename BestBinaryNumericSplit<FitnessFunction>::AuxiliarySplitInfo;

  /**
   * Check if we can split a node.  If we can split a node in a way that
   * improves on bestGain, then we return the improved gain.  Otherwise we
   * return the value DBL_MAX.
   *
   * This overload is used only for classification.
   *
   * @param bestGain Best gain seen so far (we'll only split if we find gain
   *      better than this).
   * @param data The dimension of data points to check for a split in.
   * @param numCategories Number of categories in the categorical data.
   * @param labels Labels for each point.
   * @param numClasses Number of classes in the dataset.
   * @param weights Weights associated with labels.
   * @param minLeafSize min number of points in a leaf node for
   *      splitting.
   * @param minGainSplit min  gain split.
   * @param splitInfo Stores split information on a successful split. A
   * vector of size J, where J is the number of categories. splitInfo[k]
   * is zero if category k is assigned to the left child, and otherwise
   * it is one if assigned to the right.
   * @param aux (ignored)
   */
  template<bool UseWeights, typename VecType, typename LabelsType,
          typename WeightVecType>
  static double SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const size_t numCategories,
      const LabelsType& labels,
      const size_t numClasses,
      const WeightVecType& weights,
      const size_t minLeafSize,
      const double minGainSplit,
      arma::vec& splitInfo,
      AuxiliarySplitInfo& aux);

  /**
   * Check if we can split a node.  If we can split a node in a way that
   * improves on bestGain, then we return the improved gain.  Otherwise we
   * return the value DBL_MAX.
   *
   * Overload for regression. As mentioned above, the result of Fisher only
   * applies under l₂ loss, and thus this overload is used only for regression
   * with MSEGain.
   *
   * @param bestGain Best gain seen so far (we'll only split if we find gain
   *      better than this).
   * @param data The dimension of data points to check for a split in.
   * @param numCategories Number of categories in the categorical data.
   * @param responses Responses for each point.
   * @param weights Weights associated with responses.
   * @param minLeafSize min number of points in a leaf node for
   *      splitting.
   * @param minGainSplit min  gain split.
   * @param splitInfo Stores split information on a successful split.
   *
   * @param splitInfo Stores split information on a successful split. A
   * vector of size J, where J is the number of categories. splitInfo[k]
   * is zero if category k is assigned to the left child, and otherwise
   * it is one if assigned to the right.
   * @param aux (ignored)
   * @param fitnessFunction The FitnessFunction object instance. It it used to
   *      evaluate the gain for the split.
   */
  template<bool UseWeights, typename VecType, typename ResponsesType,
           typename WeightVecType>
  static double SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const size_t numCategories,
      const ResponsesType& responses,
      const WeightVecType& weights,
      const size_t minLeafSize,
      const double minGainSplit,
      arma::vec& splitInfo,
      AuxiliarySplitInfo& aux,
      FitnessFunction& fitnessFunction);

  /**
   * In the case that a split was found, returns the number of children
   * of the split. Otherwise if there was no split, returns zero. A binary
   * split always has two children.
   *
   * @param splitInfo Auxiliary information for the split. A vector
   * of size J, where J is the number of categories. splitInfo[k]
   * is zero if category k is assigned to the left child, and otherwise
   * it is one if assigned to the right.
   * @param * (aux) Auxiliary information for the split (Unused).
   */
  static size_t NumChildren(const arma::vec& splitInfo,
                            const AuxiliarySplitInfo& /* aux */)
  {
      return splitInfo.n_elem == 0 ? 0 : 2;
  }

  /**
   *
   * In the case that a split was found, given a point, calculates
   * the index of the child it should go to. Otherwise if there was
   * no split, returns SIZE_MAX.
   *
   * @param point the Point to use.
   * @param splitInfo Auxiliary information for the split. A vector
   * of size J, where J is the number of categories. splitInfo[k] is
   * zero if category Cₖ is assigned to the left child, and otherwise
   * it is one if Cₖ is assigned to the right.
   * @param * (aux) Auxiliary information for the split (Unused).
   */
  template<typename ElemType>
  static size_t CalculateDirection(
      const ElemType& point,
      const arma::vec& splitInfo,
      const AuxiliarySplitInfo& /* aux */)
  {
    return splitInfo.n_elem == 0 ? SIZE_MAX : (size_t) splitInfo[point];
  }

 private:
  /**
   * Auxiliary for SplitIfBetter in the multi-class setting. Recursively
   * enumerates all partitions (Gₗ, Gᵣ) of categories C₀, ..., Cⱼ₋₁, and
   * computes the gain for each one, where samples with vₖ ∈ Gₗ are assigned
   * to the left tree Tₗ and those with vₖ ∈ Gᵣ to the right tree Tᵣ.
   *
   * In the case that a better split is found, bestFoundGain is updated with
   * the gain value and splitInfo is updated with the corresponding partition.
   *
   * @param labels -- Labels for each point.
   * @param numClasses -- Number of classes in the dataset.
   * @param numCategories Number of categories in the categorical data.
   * @param bestFoundGain -- The best gain found thus far. Updated if
   *    and when a better split is found.
   * @param categorySamples -- Map from category Cⱼ to the samples whose
   *    categorical value for variable vₖ is Cⱼ. Column j is for Cⱼ.
   * @param categories -- J dimensional vector used to maintain the
   *    current partition of the categories.
   * @param splitInfo -- Stores split information on a successful split. A
   *    vector of size J, where J is the number of categories. splitInfo[k]
   *    is zero if category k is assigned to the left child, and otherwise
   *    it is one if assigned to the right.
   * @param classCounts -- mx2 matrix, where m = numClasses, used to compute
   *    the gain with the FitnessFunction. All zero initially.
   * @param totalLeft, totalRight -- Number of samples assigned
   *    to the left and right subtrees respectively. Initialized to zero.
   * @param k -- Index of the current category being assigned.
   *    Initialized value is zero.
   */
  template<typename VecType, typename LabelsType>
  static bool PartitionSplit(
      const VecType& data,
      const LabelsType& labels,
      const size_t numCategories,
      const size_t numClasses,
      double& bestFoundGain,
      arma::SpMat<short>& categorySamples,
      arma::uvec& categories,
      arma::vec& splitInfo,
      arma::Mat<size_t>& classCounts,
      size_t totalLeft = 0,
      size_t totalRight = 0,
      size_t k = 0);

  /**
   * Auxiliary for SplitIfBetter in the multi-class setting. Recursively
   * enumerates all partitions (Gₗ, Gᵣ) of categories C₀, ..., Cⱼ₋₁, and
   * computes the gain for each one, where samples with vₖ ∈ Gₗ are assigned
   * to the left tree Tₗ and those with vₖ ∈ Gᵣ to the right tree Tᵣ.
   *
   * In the case that a better split is found, bestFoundGain is updated with
   * the gain value and splitInfo is updated with the corresponding partition.
   *
   * This overload is used to compute the partition using weights, that is
   * when the template variable UseWeights is true.
   *
   * @param labels -- Labels for each point.
   * @param numCategories Number of categories in the categorical data.
   * @param numClasses -- Number of classes in the dataset.
   * @param weights -- Weights associated with labels.
   * @param totalWeight -- Sum of weights.
   * @param bestFoundGain -- The best gain found thus far. Updated if
   *    and when a better split is found.
   * @param categorySamples -- Map from category Cⱼ to the samples whose
   *    categorical value for variable vₖ is Cⱼ. Column j is for Cⱼ.
   * @param categories -- J dimensional vector used to maintain the
   *    current partition of the categories.
   * @param splitInfo -- Stores split information on a successful split. A
   *    vector of size J, where J is the number of categories. splitInfo[k]
   *    is zero if category k is assigned to the left child, and otherwise
   *    it is one if assigned to the right.
   * @param classWeightSums -- mx2 matrix, where m = numClasses, used to
   *    compute the gain with the FitnessFunction. All zero initially.
   * @param totalLeftWeight, totalRightWeight -- Weight assigned to the left
   *    and right subtree respectively. Initialized to zero.
   * @param k -- Index of the current category being assigned.
   *    Initialized value is zero.
   */
  template<typename VecType, typename LabelsType, typename WeightVecType>
  static bool PartitionSplit(
      const VecType& data,
      const LabelsType& labels,
      const size_t numCategories,
      const size_t numClasses,
      const WeightVecType& weights,
      const double totalWeight,
      double& bestFoundGain,
      arma::SpMat<short>& categorySamples,
      arma::uvec& categories,
      arma::vec& splitInfo,
      arma::mat& classWeightSums,
      double totalLeftWeight = 0.0,
      double totalRightWeight = 0.0,
      size_t k = 0);
};

} // namespace mlpack

// Include implementation.
#include "best_binary_categorical_split_impl.hpp"

#endif
