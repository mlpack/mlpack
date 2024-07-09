/**
 * @file methods/xgboost/xgbtree/xgbtree.hpp
 * @author Abhimanyu Dayal
 *
 * A decision tree learner specific to XGB. Its behavior can be controlled 
 * via template arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_XGBTREE_HPP
#define MLPACK_METHODS_XGBTREE_HPP

#include "node.hpp"
#include "xgbtree.hpp"

// Defined within the mlpack namespace.
namespace mlpack {

//! Construct and train without weight.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename LabelsType>
XGBTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             NoRecursion>::XGBTree(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    LabelsType labels,
    const size_t numClasses,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector)
{
  
}




}; // mlpack

#endif