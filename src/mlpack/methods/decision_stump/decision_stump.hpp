/**
 * @file decision_stump.hpp
 * @author Udit Saxena
 * 
 * Defintion of decision stumps.
 */

#ifndef _MLPACK_METHODS_DECISION_STUMP_HPP
#define _MLPACK_METHODS_DECISION_STUMP_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace decision_stump {
/*
 * This class implements a decision stump. It constructs a single level
 * decision tree, i.e. a decision stump. It uses entropy to decided splitting
 * ranges.
 *
 */
template <typename MatType = arma::mat>
class DecisionStump
{
 public:
  /*
  Constructor. Train on the provided data. Generate a decision stump
  from data. 

  @param: data - Input, training data.
  @param: labels - Labels of data.
  @param: classes - number of distinct classes in labels.
  @param: inpBucketSize - minimum size of bucket when splitting.
   */
  DecisionStump(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t classes, 
                size_t inpBucketSize);

  /*
  Classification function. After training, classify test, and put the 
  predicted classes in predictedLabels.

  @param: test - testing data or data to classify. 
  @param: predictedLabels - vector to store the predicted classes after
                            classifying test
   */
  void Classify(const MatType& test, arma::Row<size_t>& predictedLabels);

 private:
  /* Stores the number of classes.*/
  size_t numClass; 
  
  /* Stores the default class. Provided for handling missing attribute values.*/
  size_t defaultClass;
  
  /* Stores the value of the attribute on which to split.*/
  int splitCol;
  
  /* Flag value for distinct input class labels.*/
  int oneClass; 
  
  /* Size of bucket while determining splitting criterion.*/
  size_t bucketSize;
  
  /* Stores the class labels for the input data*/
  arma::Row<size_t> classLabels;
  
  /* Stores the splitting criterion after training.*/
  arma::mat split;
  
  /* 
  Sets up attribute as if it were splitting on it and 
  finds entropy when splitting on attribute.

  @param: attribute - a row from the training data, which might be a
                      candidate for the splitting attribute.
  */
  double SetupSplitAttribute(const arma::rowvec& attribute);

  /* 
  After having decided the attribute on which to split, 
  train on that attribute.

  @param: attribute - attribute is the attribute decided by the constructor
                      on which we now train the decision stump.
   */
  template <typename rType> void TrainOnAtt(const arma::rowvec& attribute);

  /* After the "split" matrix has been set up, 
     merging ranges with identical class labels.
   */
  void MergeRanges();

  /* 
  Used to count the most frequently occurring element in subCols.

  @param: subCols - the vector in which to find the most frequently 
                    occurring element.  
   */
  template <typename rType> rType CountMostFreq(const arma::Row<rType>& subCols);
 
  /* 
  Returns 1 if all the values of featureRow are not same.

  @param: featureRow - the attribute which is checked so that it 
                       does not have identical values. 
  */
  template <typename rType> int isDistinct(const arma::Row<rType>& featureRow);

  /* 
  Calculating Entropy of attribute.

  @param: attribute - the attribute of which we calculate the entropy.
  @param: labels - corresponding labels of the attribute.
  */
  double CalculateEntropy(const arma::rowvec& attribute, 
                          const arma::rowvec& labels);

  
};

}; //namespace decision_stump
}; //namespace mlpack

#include "decision_stump_impl.cpp"

#endif