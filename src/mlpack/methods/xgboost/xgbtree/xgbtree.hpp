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

#include <mlpack/core.hpp>
#include "node.hpp"
#include "dt_prereq.hpp"
#include "../feature_importance.hpp"

// Defined within the mlpack namespace.
namespace mlpack {

class XGBTree
{
  public:

  XGBTree() { /*Nothing to do*/ }

  template<typename MatType>
  XGBTree(MatType& data,
          arma::mat& residue,
          const size_t numClasses,
          const size_t minimumLeafSize,
          const double minimumGainSplit,
          const size_t maximumDepth,
          FeatureImportance* featImp)
  {
    this.numClasses = numClasses;
    nodes.resize(numClasses);

    Train(data, residue, minimumLeafSize, minimumGainSplit, maximumDepth, featImp);
  }

  template<typename MatType>
  void Train(MatType& data,
             arma::mat& residue,
             const size_t numClasses,
             const size_t minimumLeafSize,
             const double minimumGainSplit,
             const size_t maximumDepth)
  {
    this.numClasses = numClasses;
    nodes.resize(numClasses);

    Train(data, residue, minimumLeafSize, minimumGainSplit, maximumDepth, featImp);
  }

  template<typename MatType>
  void Train(MatType& data,
             arma::mat& residue,
             const size_t minimumLeafSize,
             const double minimumGainSplit,
             const size_t maximumDepth,
             FeatureImportance* featImp)
  {
    for (size_t i = 0; i < numClasses; ++i)
    {
      Node* node = new Node(data, residue.col(i), 
        minimumLeafSize, minimumGainSplit, maximumDepth, featImp);
      nodes[i] = node;
    }
  }

  template<typename VecType>
  void Classify(VecType& point,
                arma::rowvec& rawScores)
  {
    rawScores.clear();
    rawScores.resize(numClasses);

    for (size_t i = 0; i < numClasses; ++i)
      rawScores(i) = nodes[i]->Predict(point);
  }

  template<typename MatType>
  void Classify(MatType& data,
                arma::mat& rawScores)
  {
    rawScores.clear();
    rawScores.resize(numClasses, data.n_cols);

    for (size_t i = 0l i < data.n_cols; ++i)
      Classify(data.col(i), rawScores.col(i));
  }

  private:

  //! Vector of Node pointers of length numClasses. One Node pointer for each class.
  vector<Node*> nodes; 

  //! Number of classes.
  size_t numClasses;

  //! Stores the number of times a particular feature was used to split
  map<size_t, size_t> featureFrequency;

  //! Stores the net gain provided by a particular feature for splitting
  map<size_t, double> featureCover;

};

}; // mlpack

#endif