/**
 * @file decision_stump_impl.hpp
 * @author Udit Saxena
**/

#ifndef _MLPACK_METHODS_DECISION_STUMP_IMPL_HPP
#define _MLPACK_METHODS_DECISION_STUMP_IMPL_HPP

#include "decision_stump.hpp"

#include <set>
#include <algorithm>

namespace mlpack {
namespace decision_stump {
/*
  Constructor. Train on the provided data. Generate a decision stump
  from data. 

  @param: data - Input, training data.
  @param: labels - Labels of data.
  @param: classes - number of distinct classes in labels.
  @param: inpBucketSize - minimum size of bucket when splitting.
 */
template<typename MatType>
DecisionStump<MatType>::DecisionStump(const MatType& data,
                                      const arma::Row<size_t>& labels,
                                      const size_t classes,
                                      size_t inpBucketSize)
{
  classLabels = labels + arma::zeros<arma::Row<size_t> >(labels.n_elem);
  
  numClass = classes;
  bucketSize = inpBucketSize;

  /* Check whether the input labels are not all identical. */
  if ( !isDistinct<size_t>(classLabels) )
  {
    // If the classLabels are all identical, 
    // the default class is the only class set. 
    oneClass = 1;
    defaultClass = classLabels(0); 
  }

  else
  {
    // If classLabels are not all identical
    // proceed for training

    oneClass = 0;
    int bestAtt=-1,i,j;
    double entropy,bestEntropy=DBL_MAX; 

    // Set the default class to handle attribute values which are 
    // not present in the training data. 
    defaultClass = CountMostFreq<size_t>(classLabels);

    for (i = 0;i < data.n_rows; i++)
    {
      // going through each attribute of data.
      if (isDistinct<double>(data.row(i)))
      {
        // for each attribute with non-identical values, 
        // treat it as a potential splitting attribute
        // and calculate entropy if split on it.
        entropy=SetupSplitAttribute(data.row(i));
    
        // finding the attribute with the bestEntropy
        // so that the gain is max.
        if (entropy < bestEntropy)
        {
          bestAtt = i;
          bestEntropy = entropy;
        }

      }
    }
    splitCol = bestAtt;

    // once the splitting column/attribute has been decided, 
    // train on it.
    TrainOnAtt<double>(data.row(splitCol));
  }
}

/*
  Classification function. After training, classify test, and put the 
  predicted classes in predictedLabels.

  @param: test - testing data or data to classify. 
  @param: predictedLabels - vector to store the predicted classes after
                            classifying test
 */
template<typename MatType>
void DecisionStump<MatType>::Classify(const MatType& test,
                                      arma::Row<size_t>& predictedLabels)
{
  int i,j,flag;
  double val,testval;
  if ( !oneClass )
  {
    for (i = 0; i < test.n_cols; i++)
    {
      j = 0;
      flag = 0;

      while ((j < split.n_rows) && (!flag))
      {
        if(val < split(j,0) && (!j))
        {
          predictedLabels(i) = split(0,1);
          flag = 1;
        }
        else if (val >= split(j,0))
        {
          if(j == split.n_rows - 1)
          {
            predictedLabels(i) = split(split.n_rows - 1, 1);
            flag = 1;
          }
          else if (val < split(j+1,0))
          {
            predictedLabels(i) = split(j,1);
            flag = 1;
          }
        }
        j++;
      }
    }
  }
  else
  {
    for (i = 0;i < test.n_cols;i++)
      predictedLabels(i)=defaultClass;
  }

}

/* 
  Sets up attribute as if it were splitting on it and 
  finds entropy when splitting on attribute.

  @param: attribute - a row from the training data, which might be a
                      candidate for the splitting attribute.
 */
template <typename MatType>
double DecisionStump<MatType>::SetupSplitAttribute(const arma::rowvec& attribute)
{
  int i, count, begin, end;
  double entropy = 0.0;

  // sorting the attribute, for calculating splitting ranges
  arma::rowvec sortedAtt = arma::sort(attribute);

  // storing the indexes of the sorted attribute to build 
  // a vector of sorted labels.
  // this sort is stable.
  arma::uvec sortedIndexAtt = arma::stable_sort_index(attribute.t());

  // vector of sorted labels
  arma::Row<size_t> sortedLabels(attribute.n_elem,arma::fill::zeros);
  
  for (i = 0; i < attribute.n_elem; i++)
    sortedLabels(i) = classLabels(sortedIndexAtt(i));

  arma::rowvec subColLabels;
  arma::rowvec subColAtts;

  i = 0;
  count = 0;

  // this splits the sorted into buckets of size >= inpBucketSize
  while (i < sortedLabels.n_elem)
  {
    count++;
    if (i == sortedLabels.n_elem - 1)
    {
      begin = i - count + 1;
      end = i;

      subColLabels = sortedLabels.cols(begin, end) + 
              arma::zeros<arma::rowvec>((sortedLabels.cols(begin, end)).n_elem);

      subColAtts = sortedAtt.cols(begin, end) + 
              arma::zeros<arma::rowvec>((sortedAtt.cols(begin, end)).n_elem);

      entropy += CalculateEntropy(subColAtts, subColLabels);
      i++;
    }
    else if( sortedLabels(i) != sortedLabels(i + 1) )
    {
      if (count < bucketSize) 
      {
        begin = i - count + 1;
        end = begin + bucketSize - 1;
        
        if ( end > sortedLabels.n_elem - 1)
          end = sortedLabels.n_elem - 1;
      }
      else
      {
        begin = i - count + 1;
        end = i;
      }

      subColLabels = sortedLabels.cols(begin, end) + 
              arma::zeros<arma::rowvec>((sortedLabels.cols(begin, end)).n_elem);

      subColAtts = sortedAtt.cols(begin, end) + 
              arma::zeros<arma::rowvec>((sortedAtt.cols(begin, end)).n_elem);

      // now using subColLabels and subColAtts to calculate entropuy
      entropy += CalculateEntropy(subColAtts, subColLabels);

      i = end + 1;
      count = 0;

    }
    else
      i++;
  }
  return entropy;
}

/* 
  After having decided the attribute on which to split, 
  train on that attribute.

  @param: attribute - attribute is the attribute decided by the constructor
                      on which we now train the decision stump.
 */
template <typename MatType>
template <typename rType>
void DecisionStump<MatType>::TrainOnAtt(const arma::rowvec& attribute)
{
  int i, count, begin, end;

  arma::rowvec sortedSplitAtt = arma::sort(attribute);
  arma::uvec sortedSplitIndexAtt = arma::stable_sort_index(attribute.t());
  arma::Row<size_t> sortedLabels(attribute.n_elem,arma::fill::zeros);
  arma::mat tempSplit;

  for (i = 0; i < attribute.n_elem; i++)
    sortedLabels(i) = classLabels(sortedSplitIndexAtt(i));
  
  arma::rowvec subCols;
  rType mostFreq;
  i = 0;
  count = 0;
  while (i < sortedLabels.n_elem)
  {
    count++;
    if (i == sortedLabels.n_elem - 1)
    {
      begin = i - count + 1;
      end = i;

      subCols = sortedLabels.cols(begin, end) + 
              arma::zeros<arma::rowvec>((sortedLabels.cols(begin, end)).n_elem);

      mostFreq = CountMostFreq<double>(subCols);

      tempSplit << sortedSplitAtt(begin)<< mostFreq << arma::endr;
      split = arma::join_cols(split, tempSplit);

      i++;
    }
    else if( sortedLabels(i) != sortedLabels(i + 1) )
    {
      if (count < bucketSize) // test for differevalues of bucketSize, especially extreme cases. 
      {
        begin = i - count + 1;
        end = begin + bucketSize - 1;
        
        if ( end > sortedLabels.n_elem - 1)
          end = sortedLabels.n_elem - 1;
      }
      else
      {
        begin = i - count + 1;
        end = i;
      }
      subCols = sortedLabels.cols(begin, end) + 
              arma::zeros<arma::rowvec>((sortedLabels.cols(begin, end)).n_elem);

      // finding the most freq element in subCols so as to assign a label to the
      // bucket of subCols

      mostFreq = CountMostFreq<double>(subCols);

      tempSplit << sortedSplitAtt(begin)<< mostFreq << arma::endr;
      split = arma::join_cols(split, tempSplit);

      i = end + 1;
      count = 0;
    }
    else
      i++;
  }

  // now trimming the split matrix so that buckets one after the after 
  // which point to the same classLabel are merged as one big bucket.
  MergeRanges();
}

/* After the "split" matrix has been set up, 
     merging ranges with identical class labels.
 */
template <typename MatType>
void DecisionStump<MatType>::MergeRanges()
{
  int i;
  for (i = 1;i < split.n_rows; i++)
  {
    if (split(i,1) == split(i-1,1))
    {
      // remove this row, as it has the same label as
      // the previous bucket.
      split.shed_row(i);
      // go back to previous row.
      i--;
    }
  }
}

template <typename MatType>
template <typename rType>
rType DecisionStump<MatType>::CountMostFreq(const arma::Row<rType>& subCols)
{
  // sort subCols for easier processing.
  arma::Row<rType> sortCounts = arma::sort(subCols);
  rType element;
  int count = 0, localCount = 0,i;

  // an O(n) loop which counts the most frequent element in sortCounts
  for (i = 0; i < sortCounts.n_elem ; ++i)
  {
    if (i == sortCounts.n_elem - 1)
    {
      if (sortCounts(i-1) == sortCounts(i))
      {
        // element = sortCounts(i-1);
        localCount++;
      }
      else
      if (localCount > count)
        count = localCount;
    }
    else if (sortCounts(i) != sortCounts(i+1))
    {
      localCount = 0;
      count++;
    }
    else
    {
      localCount++;
      if (localCount > count)
      {
        count = localCount;
        if(localCount == 1)
          element = sortCounts(i);
      }
    }
  }
  return element;
}

/* 
  Returns 1 if all the values of featureRow are not same.

  @param: featureRow - the attribute which is checked so that it 
                       does not have identical values. 
 */
template <typename MatType>
template <typename rType>
int DecisionStump<MatType>::isDistinct(const arma::Row<rType>& featureRow)
{
  if (featureRow.max()-featureRow.min() > 0)
    return 1;
  else
    return 0;
}

/* 
  Calculating Entropy of attribute.

  @param: attribute - the attribute of which we calculate the entropy.
  @param: labels - corresponding labels of the attribute.
 */
template<typename MatType>
double DecisionStump<MatType>::CalculateEntropy(const arma::rowvec& attribute,
                                                const arma::rowvec& labels)
{
  int i,j,count;
  double entropy=0.0;

  arma::rowvec uniqueAtt = arma::unique(attribute);
  arma::rowvec uniqueLabel = arma::unique(labels);
  arma::Row<size_t> numElem(uniqueAtt.n_elem,arma::fill::zeros); 
  arma::Mat<size_t> entropyArray(uniqueAtt.n_elem,numClass,arma::fill::zeros); 
  
  // populating entropyArray and numElem, they are to be used as 
  // helpers to calculate entropy
  for (j = 0;j < uniqueAtt.n_elem; j++)
  {
    for (i = 0; i < attribute.n_elem; i++)
    {
      if (uniqueAtt[j] == attribute[i])
      {
        entropyArray(j,labels(i))++;
        numElem(j)++;
      }
    }
  }

  double p1, p2, p3;
  for ( j = 0; j < uniqueAtt.size(); j++ )
  {
    p1 = ((double)numElem(j) / attribute.n_elem);

    for ( i = 0; i < numClass; i++)
    {
      p2 = ((double)entropyArray(j,i) / numElem(j));
      
      if(p2 == 0)
        p3 = 0;
      else
        p3 = (  p2 * log2(p2) );

      entropy+=( p1 * p3 );
    }
  }

  return entropy;
}


}; // namespace decision_stump
}; // namespace mlpack

#endif