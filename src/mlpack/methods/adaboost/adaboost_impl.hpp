/*
 * @file adaboost_impl.hpp
 * @author Udit Saxena
 *
 * Implementation of the Adaboost class
 *
 *  @code
 *  @article{Schapire:1999:IBA:337859.337870,
 *  author = {Schapire, Robert E. and Singer, Yoram},
 *  title = {Improved Boosting Algorithms Using Confidence-rated Predictions},
 *  journal = {Mach. Learn.},
 *  issue_date = {Dec. 1999},
 *  volume = {37},
 *  number = {3},
 *  month = dec,
 *  year = {1999},
 *  issn = {0885-6125},
 *  pages = {297--336},
 *  numpages = {40},
 *  url = {http://dx.doi.org/10.1023/A:1007614523901},
 *  doi = {10.1023/A:1007614523901},
 *  acmid = {337870},
 *  publisher = {Kluwer Academic Publishers},
 *  address = {Hingham, MA, USA},
 *  keywords = {boosting algorithms, decision trees, multiclass classification, output coding
 *  }
 *  @endcode
 *
}
 */

#ifndef _MLPACK_METHODS_ADABOOST_ADABOOST_IMPL_HPP
#define _MLPACK_METHODS_ADABOOST_ADABOOST_IMPL_HPP

#include "adaboost.hpp"

namespace mlpack {
namespace adaboost {
/**
 *  Constructor. Currently runs the Adaboost.mh algorithm
 *  
 *  @param data Input data
 *  @param labels Corresponding labels
 *  @param iterations Number of boosting rounds 
 *  @param classes Number of classes in labels
 *  @param other Weak Learner, which has been initialized already
 */
template<typename MatType, typename WeakLearner>
Adaboost<MatType, WeakLearner>::Adaboost(const MatType& data, 
        const arma::Row<size_t>& labels, int iterations, 
        size_t classes, const WeakLearner& other)
{
  // note: put a fail safe for the variable 'classes' or 
  // remove it entirely by using unique function.
  int i, j, k;
  double rt, alphat = 0.0, zt;
  
  // To be used for prediction by the Weak Learner for prediction.
  arma::Row<size_t> predictedLabels(labels.n_cols);
  
  // Use tempData to modify input Data for incorporating weights.
  MatType tempData(data);
  
  // Build the classification Matrix yt from labels
  arma::mat yt(predictedLabels.n_cols, classes);
  
  // Build a classification matrix of the form D(i,l)
  // where i is the ith instance
  // l is the lth class.
  buildClassificationMatrix(yt, labels);
  
  // ht(x), to be loaded after a round of prediction every time the weak
  // learner is run, by using the buildClassificationMatrix function
  arma::mat ht(predictedLabels.n_cols, classes);

  // This matrix is a helper matrix used to calculate the final hypothesis.
  arma::mat sumFinalH(predictedLabels.n_cols, classes);
  sumFinalH.fill(0.0);
  
  // load the initial weights into a 2-D matrix
  const double initWeight = 1 / (data.n_cols * classes);
  arma::mat D(data.n_cols, classes);
  D.fill(initWeight);

  // Weights are to be compressed into this rowvector
  // for focussing on the perceptron weights.
  arma::rowvec weights(predictedLabels.n_cols);

  // This is the final hypothesis.
  arma::rowvec finalH(predictedLabels.n_cols);

  // now start the boosting rounds
  for (i = 0; i < iterations; i++)
  {
    // Initialized to zero in every round.
    rt = 0.0; 
    zt = 0.0;
    
    // Build the weight vectors
    buildWeightMatrix(D, weights);
    
    // call the other weak learner and train the labels.
    WeakLearner w(other, tempData, weights, labels);
    w.Classify(tempData, predictedLabels);

    //Now from predictedLabels, build ht, the weak hypothesis
    buildClassificationMatrix(ht, predictedLabels);

    // Now, start calculation of alpha(t) using ht
    
    // begin calculation of rt

    for (j = 0;j < ht.n_rows; j++)
    {
      for (k = 0;k < ht.n_cols; k++)
        rt += (D(j,k) * yt(j,k) * ht(j,k));
    }

    // end calculation of rt

    alphat = 0.5 * log((1 + rt) / (1 - rt));

    // end calculation of alphat
    
    // now start modifying weights

    for (j = 0;j < D.n_rows; j++)
    {
      for (k = 0;k < D.n_cols; k++)
      {  
        // we calculate zt, the normalization constant
        zt += D(j,k) * exp(-1 * alphat * yt(j,k) * ht(j,k));
        D(j,k) = D(j,k) * exp(-1 * alphat * yt(j,k) * ht(j,k));

        // adding to the matrix of FinalHypothesis 
        sumFinalH(j,k) += (alphat * ht(j,k));
      }
    }

    // normalization of D

    D = D / zt;
  
  }

  // Iterations are over, now build a strong hypothesis
  // from a weighted combination of these weak hypotheses.
  
  // This step of storing it in a temporary row vector can be improved upon ? 
  arma::rowvec tempSumFinalH;
  arma::uword max_index;
  for (i = 0;i < sumFinalH.n_rows; i++)
  {
    tempSumFinalH = sumFinalH.row(i);
    tempSumFinalH.max(max_index);
    finalH(i) = max_index;
  }

  //finalH is the final hypothesis.
}

/**
 *  This function helps in building a classification Matrix which is of 
 *  form: 
 *  -1 if l is not the correct label
 *  1 if l is the correct label
 *
 *  @param t The classification matrix to be built
 *  @param l The labels from which the classification matrix is to be built.
 */
template <typename MatType, typename WeakLearner>
void Adaboost<MatType, WeakLearner>::buildClassificationMatrix(
                                     arma::mat& t, const arma::Row<size_t>& l)
{
  int i, j;

  for (i = 0;i < t.n_rows; i++)
  {
    for (j = 0;j < t.n_cols; j++)
    {
      if (j == l(i))
        t(i,j) = 1.0;
      else
        t(i,j) = -1.0;
    }
  }
}

/**
 *  This function helps in building the Weight Distribution matrix
 *  which is updated during every iteration. It calculates the 
 *  "difficulty" in classifying a point by adding the weights for all 
 *  instances, using D.
 *  
 *  @param D The 2 Dimensional weight matrix from which the weights are
 *            to be calculated.
 *  @param weights The output weight vector.
 */
template <typename MatType, typename WeakLearner>
void Adaboost<MatType, WeakLearner>::buildWeightMatrix(
                                     const arma::mat& D, arma::rowvec& weights)
{
  int i, j;
  weights.fill(0.0);

  for (i = 0;i < D.n_rows; i++)
  {
    for (j = 0;j < D.n_cols; j++)
      weights(i) += D(i,j);
  }
}

} // namespace adaboost
} // namespace mlpack

#endif