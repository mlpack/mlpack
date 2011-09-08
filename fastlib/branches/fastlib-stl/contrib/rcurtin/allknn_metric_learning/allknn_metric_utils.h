/***
 * @file allknn_metric_utils.h
 * @author Ryan Curtin
 *
 * Our objective is to learn a metric to optimize the performance of
 * all-k-nearest-neighbors.  The specific thing we are seeking is the matrix A
 * in the distance calculation d(x_i, x_j) = x_i^T * A * x_j; in this method we
 * have restricted A to being diagonal.
 *
 * A user passes in the dataset which is to be analyzed with AllkNN, an output
 * file to store the weights, and optionally a file of initial weights to start
 * from (if, for instance, a better set of weights is already known).  The
 * number of iterations performed, the number of neighbors k, and the learning
 * rate alpha can all be specified.
 *
 * This code uses the old-style FASTLIB FX system (which will need to be
 * changed).
 */

#ifndef __MLPACK_CONTRIB_RCURTIN_ALLKNN_METRIC_UTILS_H
#define __MLPACK_CONTRIB_RCURTIN_ALLKNN_METRIC_UTILS_H

#include <fastlib/fastlib.h>
#include <armadillo>
#include <map>

namespace mlpack {
namespace allknn {
namespace metric {

/***
 * Take a majority vote among the labels of neighbors to determine the class of
 * a point.  The only parameter is the list of sorted neighbor labels.  The
 * order of labels will affect the vote -- in the event of a tie, the label with
 * lower indices (closer neighbors) will "win".
 *
 * @param neighbor_labels Label of neighbors; the first will be the closest.
 */
int ClassifyKNN(arma::vec& neighbor_labels);

/***
 * Given a new weighting vector, estimate the new classification score.  The
 * resultant neighbors list and distance list from the AllkNN class are taken as
 * parameters.  The returned integer is the number of correctly classified
 * points.
 *
 * @param input Matrix of input points
 * @param neighbors List of calculated neighbors (length n_points * knns)
 * @param distances List of calculated distances (length n_points * knns)
 * @param labels List of correct classes of points (length n_points)
 * @param knns Value of k used in AllkNN calculation
 * @param score Number of neighbors to use for scoring ( <= knns )
 * @return Number of correctly classified points
 */
int EstimateScores(arma::mat& input,
                   arma::Col<size_t>& neighbors,
                   arma::vec& labels,
                   int knns,
                   int score);

/***
 * Given a list of calculated neighbors and their associated labels, evaluate
 * the accuracy of the calculated neighbors.  The vector of neighbors should be
 * of the same format as the output of the AllkNN class.
 *
 * @param neighbors List of calculated neighbors (length n_points * knns)
 * @param knns k used in AllkNN calculation
 * @param score Number of neighbors to use for scoring ( <= knns )
 * @param labels List of correct classes of points (length n_points)
 * @return Number of correctly classified points
 */
int EvaluateCorrect(arma::Col<size_t>& neighbors,
                    int knns,
                    int score,
                    arma::vec& labels);

/***
 * Given a list of calculated neighbors and their associated labels, evaluate
 * the accuracy of the calculated neighbors of a single index.  The vector of
 * neighbors should be of the same format as the output of the AllkNN class.
 *
 * @param neighbors List of calculated neighbors (length n_points * knns)
 * @param knns k used in AllkNN calculation
 * @param score Number of neighbors to use for scoring ( <= knns )
 * @param labels List of correct classes of points (length n_points)
 * @param index Index of point to evaluate
 * @return Number of correctly classified points
 */
int EvaluateCorrect(arma::Col<size_t>& neighbors,
                    int knns,
                    int score,
                    arma::vec& labels,
                    int index);

/***
 * Given a list of calculated neighbors and the labels of the reference and
 * query points, evaluate the model (as in EvaluateCorrect()), but also store
 * counts of the correctly classified items in each class and the total number
 * of items in each class.
 *
 * @param neighbors List of calculated neighbors (length n_points * knns)
 * @param knns k used in AllkNN calculation
 * @param ref_labels Labels of reference points
 * @param query_labels Labels of query points
 * @param class_counts Vector to be populated with number of points in each
 *            class
 * @param class_correct Vector to be populated with number of correctly
 *            classified points in each class
 * @return Number of correctly classified points
 */
int EvaluateClassCorrect(const arma::Col<size_t>& neighbors,
                         int knns,
                         const arma::vec& ref_labels,
                         const arma::vec& query_labels,
                         arma::Col<int>& class_counts,
                         arma::Col<int>& class_correct);

}; // namespace metric
}; // namespace allknn
}; // namespace mlpack

#endif
