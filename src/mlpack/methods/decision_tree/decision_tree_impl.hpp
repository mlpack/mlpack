/**
 * @file decision_tree_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of generic decision tree class.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_DECISION_TREE_IMPL_HPP
#define MLPACK_METHODS_DECISION_TREE_DECISION_TREE_IMPL_HPP

namespace mlpack {
namespace tree {

//! Construct and train.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             ElemType,
             NoRecursion>::DecisionTree(const MatType& data,
                                        const data::DatasetInfo& datasetInfo,
                                        const arma::Row<size_t>& labels,
                                        const size_t numClasses,
                                        const size_t minimumLeafSize)
{
  // Pass off work to the Train() method.
  Train(data, datasetInfo, labels, numClasses, minimumLeafSize);
}

//! Construct and train.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             ElemType,
             NoRecursion>::DecisionTree(const MatType& data,
                                        const arma::Row<size_t>& labels,
                                        const size_t numClasses,
                                        const size_t minimumLeafSize)
{
  // Pass off work to the Train() method.
  Train(data, labels, numClasses, minimumLeafSize);
}

//! Construct, don't train.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             ElemType,
             NoRecursion>::DecisionTree(const size_t numClasses) :
    classProbabilities(numClasses),
    dimensionTypeOrMajorityClass(0)
{
  // Initialize utility vector.
  classProbabilities.fill(1.0 / (double) numClasses);
}

//! Copy another tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             ElemType,
             NoRecursion>::DecisionTree(const DecisionTree& other) :
    splitDimension(other.splitDimension),
    dimensionTypeOrMajorityClass(other.dimensionTypeOrMajorityClass),
    classProbabilities(other.classProbabilities),
    numericAux(other.numericAux),
    categoricalAux(other.categoricalAux)
{
  // Copy each child.
  for (size_t i = 0; i < other.children.size(); ++i)
    children.push_back(new DecisionTree(*other.children[i]));
}

//! Take ownership of another tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             ElemType,
             NoRecursion>::DecisionTree(DecisionTree&& other) :
    children(std::move(other.children)),
    splitDimension(other.splitDimension),
    dimensionTypeOrMajorityClass(other.dimensionTypeOrMajorityClass),
    classProbabilities(std::move(other.classProbabilities)),
    numericAux(std::move(other.numericAux)),
    categoricalAux(std::move(other.categoricalAux))
{
  // Reset the other object.
  other.classProbabilities.ones(1); // One class, P(1) = 1.
}

//! Copy another tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             ElemType,
             NoRecursion>&
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             ElemType,
             NoRecursion>::operator=(const DecisionTree& other)
{
  // Clean memory if needed.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  // Copy everything from the other tree.
  splitDimension = other.splitDimension;
  dimensionTypeOrMajorityClass = other.dimensionTypeOrMajorityClass;
  classProbabilities = other.classProbabilities;
  numericAux = other.numericAux;
  categoricalAux = other.categoricalAux;

  // Copy the children.
  for (size_t i = 0; i < other.children.size(); ++i)
    children.push_back(new DecisionTree(*other.children[i]));

  return *this;
}

//! Take ownership of another tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             ElemType,
             NoRecursion>&
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             ElemType,
             NoRecursion>::operator=(DecisionTree&& other)
{
  // Clean memory if needed.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  // Take ownership of the other tree's components.
  children = std::move(other.children);
  splitDimension = other.splitDimension;
  dimensionTypeOrMajorityClass = other.dimensionTypeOrMajorityClass;
  classProbabilities = std::move(other.classProbabilities);
  numericAux = std::move(other.numericAux);
  categoricalAux = std::move(other.categoricalAux);

  // Reset the class probabilities of the other object.
  other.classProbabilities.ones(1); // One class, P(1) = 1.

  return *this;
}

//! Clean up memory.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             ElemType,
             NoRecursion>::~DecisionTree()
{
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
}

//! Train on the given data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
template<typename MatType>
void DecisionTree<FitnessFunction,
                  NumericSplitType,
                  CategoricalSplitType,
                  ElemType,
                  NoRecursion>::Train(const MatType& data,
                                      const data::DatasetInfo& datasetInfo,
                                      const arma::Row<size_t>& labels,
                                      const size_t numClasses,
                                      const size_t minimumLeafSize)
{
  // Clear children if needed.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  // Look through the list of dimensions and obtain the gain of the best split.
  // We'll cache the best numeric and categorical split auxiliary information in
  // numericAux and categoricalAux (and clear them later if we make not split),
  // and use classProbabilities as auxiliary information.  Later we'll overwrite
  // classProbabilities to the empirical class probabilities if we do not split.
  double bestGain = FitnessFunction::Evaluate(labels, numClasses);
  size_t bestDim = datasetInfo.Dimensionality(); // This means "no split".
  for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
  {
    double dimGain;
    if (datasetInfo.Type(i) == Datatype::categorical)
      dimGain = CategoricalSplit::SplitIfBetter(bestGain, data.row(i),
          info.NumMappings(i), labels, numClasses, minimumLeafSize,
          classProbabilities, categoricalAux);
    else if (datasetInfo.Type(i) == Datatype::numeric)
      dimGain = NumericSplit::SplitIfBetter(bestGain, data.row(i), labels,
          numClasses, minimumLeafSize, classProbabilities, numericAux);

    // Was there an improvement?  If so mark that it's the new best dimension.
    if (dimGain > bestGain)
      bestDim = i;
  }

  // Did we split or not?  If so, then split the data and create the children.
  if (bestDim != datasetInfo.Dimensionality())
  {
    dimensionTypeOrMajorityClass = (size_t) datasetInfo.Type(bestDim);

    // Get the number of children we will have.
    size_t numChildren = 0;
    if (datasetInfo.Type(bestDim) == Datatype::categorical)
      numChildren = CategoricalSplit::NumChildren(classProbabilities,
          categoricalAux);
    else
      numChildren = NumericSplit::NumChildren(classProbabilities, numericAux);

    // Calculate all child assignments.
    arma::Col<size_t> childAssignments(data.n_cols);
    if (datasetInfo.Type(bestDim) == Datatype::categorical)
    {
      for (size_t j = 0; j < data.n_cols; ++j)
        childAssignments[j] = CategoricalSplit::CalculateDirection(
            classProbabilities, categoricalAux);
    }
    else
    {
      for (size_t j = 0; j < data.n_cols; ++j)
        childAssignments[j] = NumericSplit::CalculateDirection(
            classProbabilities, numericAux);
    }

    // Split into children.
    for (size_t i = 0; i < numChildren; ++i)
    {
      // Now that we have the size of the matrix we need to extract, extract it.
      MatType childPoints(data.n_rows, childCounts[i]);
      arma::Row<size_t> childLabels(childCounts[i]);
      size_t currentCol = 0;
      for (size_t j = 0; j < data.n_cols; ++j)
      {
        if (childAssignments[j] == i)
        {
          childPoints.col(currentCol) = data.col(j);
          childLabels[currentCol++] = labels[j];
        }
      }

      // Now build the child recursively.
      children.push_back(new DecisionTree(childPoints, childLabels, numClasses,
          minimumLeafSize));
    }
  }
  else
  {
    // Clear auxiliary info objects.
    numericAux = NumericSplit::AuxiliarySplitInfo<ElemType>();
    categoricalAux = CategoricalSplit::AuxiliarySplitInfo<ElemType>();

    // Calculate class probabilities because we are a leaf.
    CalculateClassProbabilities(labels);
  }
}

//! Train on the given data, assuming all dimensions are numeric.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
template<typename MatType>
void DecisionTree<FitnessFunction,
                  NumericSplitType,
                  CategoricalSplitType,
                  ElemType,
                  NoRecursion>::Train(const MatType& data,
                                      const arma::Row<size_t>& labels,
                                      const size_t numClasses,
                                      const size_t minimumLeafSize)
{
  // Clear children if needed.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  // We won't be using this.
  categoricalAux =
      CategoricalSplitType<FitnessFunction>::AuxiliarySplitInfo<ElemType>();

  // Look through the list of dimensions and obtain the best split.  We'll cache
  // the best numeric split auxiliary information in numericAux (and clear it
  // later if we don't make a split), and use classProbabilities as auxiliary
  // information.  Later we'll overwrite classProbabilities to the empirical
  // class probabilities if we do not split.
  double bestGain = FitnessFunction::Evaluate(labels, numClasses);
  size_t bestDim = data.n_rows; // This means "no split".
  for (size_t i = 0; i < data.n_rows; ++i)
  {
    double dimGain = NumericSplitType<FitnessFunction>::SplitIfBetter(bestGain,
        data.row(i), labels, numClasses, minimumLeafSize, classProbabilities,
        numericAux);

    if (dimGain > bestGain)
      bestDim = i;
  }

  // Did we split or not?  If so, then split the data and create the children.
  if (bestDim != data.n_rows)
  {
    // We know that the split is numeric.
    size_t numChildren = NumericSplit::NumChildren(classProbabilities,
        numericAux);

    // Calculate all child assignments.
    arma::Col<size_t> childAssignments(data.n_cols);
    for (size_t j = 0; j < data.n_cols; ++j)
      childAssignments[j] = NumericSplit::CalculateDirection(classProbabilities,
          numericAux);

    // Calculate counts of children in each node.
    arma::Col<size_t> childCounts(numChildren);
    childCounts.zeros();
    for (size_t j = 0; j < childAssignments.n_elem; ++j)
      childCounts[childAssignments[j]]++;

    for (size_t i = 0; i < numChildren; ++i)
    {
      // Now that we have the size of the matrix we need to extract, extract it.
      MatType childPoints(data.n_rows, childCounts[i]);
      arma::Row<size_t> childLabels(childCounts[i]);
      size_t currentCol = 0;
      for (size_t j = 0; j < data.n_cols; ++j)
      {
        if (childAssignments[j] == i)
        {
          childPoints.col(currentCol) = data.col(j);
          childLabels[currentCol++] = labels[j];
        }
      }

      // Now build the child recursively.
      children.push_back(new DecisionTree(childPoints, childLabels, numClasses,
          minimumLeafSize));
    }
  }
  else
  {
    // We won't be needing this.
    numericAux = NumericSplit::AuxiliarySplitInfo<ElemType>();

    // Calculate class probabilities because we are a leaf.
    CalculateClassProbabilities(labels);
  }
}

//! Return the class.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
template<typename VecType>
size_t DecisionTree<FitnessFunction,
                    NumericSplitType,
                    CategoricalSplitType,
                    ElemType,
                    NoRecursion>::Classify(const VecType& point) const
{
  if (children.size() == 0)
  {
    // Return cached max of probabilities.
    return dimensionTypeOrMajorityClass;
  }

  return children[CalculateDirection(point)]->Classify(point);
}

//! Return class probabilities for a given point.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplittype,
         typename ElemType,
         bool NoRecursion>
template<typename VecType>
void DecisionTree<FitnessFunction,
                  NumericSplitType,
                  CategoricalSplitType,
                  ElemType,
                  NoRecursion>::Classify(const VecType& point,
                                         size_t& prediction,
                                         arma::vec& probabilities) const
{
  if (children.size() == 0)
  {
    prediction = dimensionTypeOrMajorityClass;
    probabilities = classProbabilities;
    return;
  }

  children[CalculateDirection(point)]->Classify(point, probabilities);
}

//! Return the class for a set of points.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
template<typename MatType>
void DecisionTree<FitnessFunction,
                  NumericSplitType,
                  CategoricalSplitType,
                  ElemType,
                  NoRecursion>::Classify(const MatType& data,
                                         arma::Row<size_t>& predictions) const
{
  predictions.set_size(data.n_cols);
  if (children.size() == 0)
  {
    predictions.fill(dimensionTypeOrMajorityClass);
    return;
  }

  // Loop over each point.
  for (size_t i = 0; i < data.n_cols; ++i)
    predictions[i] = Classify(data.col(i));
}

//! Return the class probabilities for a set of points.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
template<typename MatType>
void DecisionTree<FitnessFunction,
                  NumericSplitType,
                  CategoricalSplitType,
                  ElemType,
                  NoRecursion>::Classify(const MatType& data,
                                         arma::Row<size_t>& predictions,
                                         arma::mat& probabilities) const
{
  predictions.set_size(data.n_cols);
  if (children.size() == 0)
  {
    predictions.fill(dimensionTypeOrMajorityClass);
    probabilities = arma::repmat(classProbabilities, 1, data.n_cols);
    return;
  }

  // Otherwise we have to find the right size to set the predictions matrix to
  // be.
  DecisionTree* node = children[0];
  while (node->NumChildren() != 0)
    node = &node->Child(i);
  probabilities.set_size(node.classProbabilities.n_elem, data.n_cols);

  for (size_t i = 0; i < data.n_cols; ++i)
    Classify(data.col(i), predictions[i], probabilities.unsafe_col(i));
}

//! Serialize the tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
template<typename Archive>
void DecisionTree<FitnessFunction,
                  NumericSplitType,
                  CategoricalSplitType,
                  ElemType,
                  NoRecursion>::Serialize(Archive& ar,
                                          const unsigned int /* version */)
{
  using data::CreateNVP;

  // Clean memory if needed.
  if (Archive::is_loading::value)
  {
    for (size_t i = 0; i < children.size(); ++i)
      delete children[i];
    children.clear();
  }

  // Serialize the children first.
  size_t numChildren = children.size();
  ar & CreateNVP(numChildren);
  if (Archive::is_loading::value)
  {
    children.resize(numChildren, NULL);
    for (size_t i = 0; i < numChildren; ++i)
      children[i] = new DecisionTree();
  }

  for (size_t i = 0; i < numChildren; ++i)
  {
    std::ostringstream name;
    name << "child" << i;
    ar & CreateNVP(*children[i], name.str());
  }

  // Now serialize the rest of the object.
  ar & CreateNVP(splitDimension, "splitDimension");
  ar & CreateNVP(dimensionTypeOrMajorityClass, "dimensionTypeOrMajorityClass");
  ar & CreateNVP(classProbabilities, "classProbabilities");
  ar & CreateNVP(numericAux, "numericAux");
  ar & CreateNVP(categoricalAux, "categoricalAux");
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename ElemType,
         bool NoRecursion>
template<typename VecType>
size_t DecisionTree<FitnessFunction,
                    NumericSplitType,
                    CategoricalSplitType,
                    ElemType,
                    NoRecursion>::CalculateDirection(const VecType& point) const
{
  if ((Datatype) dimensionTypeOrMajorityClass == Datatype::categorical)
    return CategoricalSplit::CalculateDirection(point, classProbabilities,
        categoricalAux);
  else
    return NumericSplit::CalculateDirection(point, classProbabilities,
        numericAux);
}

} // namespace tree
} // namespace mlpack

#endif
