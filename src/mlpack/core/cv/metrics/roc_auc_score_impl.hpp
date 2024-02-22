/**
 * @file core/cv/metrics/roc_auc_score_impl.hpp
 * @author Sri Madhan M
 *
 * Implementation of the area under Receiver Operating Characteristic curve
 * (ROC-AUC) score.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_ROCAUCSCORE_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_ROCAUCSCORE_IMPL_HPP

namespace mlpack {

template<size_t PositiveClass>
double ROCAUCScore<PositiveClass>::Evaluate(const arma::Row<size_t>& labels,
                     const  arma::mat& scores)
{
  size_t classes = arma::size(scores)[0];
  size_t n_examples = arma::size(scores)[1];


  // Checks if the no. of labels is 0.
  if (n_examples == 0)
  {
    throw std::invalid_argument(
        "ROCAUCScore::Evaluate(): "
        "number of points in input data cannot be zero");
  }

  arma::Row<size_t> class_labels = arma::unique(labels);

  // Computes AUC for the given positive score. 
  if(classes == 1 && class_labels.n_elem <= 2)
  {
      return BinaryEvaluate(labels,scores);
  }

  // Checks if the no. of unique labels is greater than the classes in score.
  if(class_labels.n_elem > classes)
  {
    throw std::invalid_argument(
        "ROCAUCScore::Evaluate(): "
        "number of unique labels in true labels cannot be greater "
        "than the no. of scores per datapoint , ROCAUCScore is undefined");
  }

  arma::mat ones(size(arma::sum(scores, 0)),arma::fill::ones);
  bool allOnes = arma::approx_equal(arma::sum(scores, 0), ones, "absdiff", 1e-5);
  
  // Checks if sum of all label probabilities is near zero.
  if(!allOnes)
  {
    throw std::invalid_argument(
        "ROCAUCScore::Evaluate(): "
        "sum of probalilies of all classifiers for a datapoint must "
        "be equal to 1, ROCAUCScore is undefined");
  }

  class_labels = arma::sort(class_labels); 
  arma::Row<size_t> binaryLabels;
  arma::rowvec binaryScore;   
  
  // Computes score for positive class and returns AUC.
  if(classes == 2)
  { 
    size_t positiveCol = class_labels(1) == PositiveClass;
    arma::rowvec binaryScore = scores.row(positiveCol);
    return BinaryEvaluate(labels, binaryScore);
  }

  // Computes the class wise AUC in a One vs Rest manner.
  double auc = 0.0;
  for(size_t class_label = 0; class_label < classes; class_label++)
  {
    // Computes positive and 0 labels for class_label. 
    if(PositiveClass != 0)
    {
      binaryLabels = arma::conv_to<arma::Row<size_t>>::from
                    (labels == class_labels(class_label)) * PositiveClass;
    }
    else
    {
      binaryLabels = arma::conv_to<arma::Row<size_t>>::from
                    (labels == class_labels(class_label));
      binaryLabels = arma::conv_to<arma::Row<size_t>>::from(binaryLabels == 0);
    }
    binaryScore = scores.row(class_labels(class_label));
    auc += BinaryEvaluate(binaryLabels, binaryScore);
  }

  // Returns average AUC.
  auc = auc / classes ;
  return auc;
}


template<size_t PositiveClass>
double ROCAUCScore<PositiveClass>::BinaryEvaluate(const arma::Row<size_t>& labels,
                                            const arma::rowvec& scores)
{
  util::CheckSameSizes(labels, scores, "ROCAUCScore::Evaluate()");

  if (labels.n_cols == 0)
  {
    throw std::invalid_argument(
        "ROCAUCScore::Evaluate(): "
        "number of points in input data cannot be zero");
  }

  // Compute labels with "1" for positive class and "0" for the other.
  arma::Col<size_t> binaryLabels = arma::conv_to<arma::Col<size_t>>::from(
      (labels == PositiveClass));

  // Converting probability scores of PositiveClass, from row to column vector.
  arma::vec colScores = arma::conv_to<arma::vec>::from(scores);

  size_t numberOfTrueLabels  = sum(binaryLabels);
  size_t numberOfFalseLabels = binaryLabels.n_rows - numberOfTrueLabels;

  // Check if only one class is given in labels.
  if (numberOfTrueLabels == 0 || numberOfFalseLabels == 0)
  {
    throw std::invalid_argument(
        "ROCAUCScore::Evaluate(): "
        "only one class is given in labels, ROCAUCScore is undefined");
  }

  // Sort labels and probabilities, using probability scores.
  arma::uvec sortedScoreIndices = arma::stable_sort_index(colScores, "descend");
  arma::Col<size_t> sortedLabels = binaryLabels(sortedScoreIndices);
  arma::vec sortedScores = colScores(sortedScoreIndices);

  // Compute indices of unique probability scores.
  arma::uword uniqueScoreIndicesLength = 0;
  arma::ucolvec uniqueScoreIndices(sortedScores.n_rows);
  for (arma::uword idx = 0; idx < sortedScores.n_rows - 1; idx++)
  {
    if (sortedScores(idx) != sortedScores(idx + 1))
    {
      uniqueScoreIndices(uniqueScoreIndicesLength++) = idx;
    }
  }
  uniqueScoreIndices(uniqueScoreIndicesLength++) = sortedScores.n_rows - 1;
  uniqueScoreIndices.resize(uniqueScoreIndicesLength);

  // Compute true positive rate, and false positive rate.
  arma::Col<size_t> cumulativeSum = arma::cumsum(sortedLabels);
  cumulativeSum = cumulativeSum(uniqueScoreIndices);

  arma::vec tpr, fpr;
  tpr = arma::conv_to<arma::vec>::from(cumulativeSum);
  fpr = 1 + uniqueScoreIndices - tpr;
  tpr /= numberOfTrueLabels;
  fpr /= numberOfFalseLabels;

  // To ensure that the (fpr, tpr) starts at (0, 0).
  tpr.insert_rows(0, 1);
  fpr.insert_rows(0, 1);
  tpr(0) = fpr(0) = 0;

  // Compute area under the curve using trapezoidal rule.
  arma::mat auc = arma::trapz(fpr, tpr);
  return auc(0, 0);
}

} // namespace mlpack

#endif
