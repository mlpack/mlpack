/***
 * @file allknn_metric_utils.cc
 * @author Ryan Curtin
 *
 * Implementation of useful utilities for metric learning and evaluation for
 * all-k-nearest-neighbors.
 */

#include "allknn_metric_utils.h"

using namespace mlpack;
using namespace mlpack::allknn;
using namespace mlpack::allknn::metric;

// Utility function for std::pair<int, double> sorting.
bool sortPairedVector(std::pair<int, double> lhs, std::pair<int, double> rhs) {
  return lhs.second < rhs.second;
}

/***
 * Take a majority vote among the labels of neighbors to determine the class of
 * a point.  The only parameter is the list of sorted neighbor labels.  The
 * order of labels will affect the vote -- in the event of a tie, the label with
 * lower indices (closer neighbors) will "win".
 *
 * @param neighbor_labels Label of neighbors; the first will be the closest.
 * @return Majority vote result (the class guess)
 */
int mlpack::allknn::metric::ClassifyKNN(arma::vec& neighbors) {
  std::map<int, int> freqs;

  for(int i = 0; i < neighbors.n_elem; i++) {
    if(freqs.count(neighbors[i]) == 0)
      freqs[neighbors[i]] = 1;
    else
      freqs[neighbors[i]] += 1;
  }

  // find most frequent neighbor
  int max = 0;
  int max_index = -1;
  for (std::map<int, int>::iterator i = freqs.begin(); i != freqs.end(); i++) {
    if(i->second > max) {
      // new max
      max_index = i->first;
      max = i->second;
    } else if (i->second == max) {
      // see which index occurs first (we'll select the index that is nearer)
      int a, b;

      for (a = 0; a < neighbors.n_elem; a++) {
        if (neighbors[a] == i->first)
          break;
      }

      for (b = 0; b < neighbors.n_elem; b++) {
        if (neighbors[b] == max_index)
          break;
      }

      if (a < b)
        max_index = i->first;
    }
  }

  return max_index;
}

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
int mlpack::allknn::metric::EstimateScores(arma::mat& input,
                                           arma::Col<index_t>& neighbors,
                                           arma::vec& labels,
                                           int knns,
                                           int score) {
  // We will not do this a particularly smart way, but instead we will just run
  // the naive calculation on the nearest {knns} neighbors as calculated with a
  // different weighting matrix, then resorting them and reclassifying.
  int correct = 0;
  for (int i = 0; i < input.n_cols; i++) {
    std::vector<std::pair<int, double> > local_neighbors;
    for (int j = 0; j < knns; j++) {
      int ind = (i * knns) + j;
      double dist = norm(input.col(i) - input.col(neighbors[ind]), 2);
      local_neighbors.push_back(std::make_pair(neighbors[ind], dist));
    }

    // Now sort our new vector by distance (since the order has likely changed).
    std::sort(local_neighbors.begin(), local_neighbors.end(), sortPairedVector);
    arma::Col<index_t> tmp(local_neighbors.size());
    for (int j = 0; j < local_neighbors.size(); j++)
      tmp(j) = local_neighbors.at(j).first;

    correct += EvaluateCorrect(tmp, knns, score, labels, i);
  }

  return correct;
}

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
int mlpack::allknn::metric::EvaluateCorrect(arma::Col<index_t>& neighbors,
                                            int knns,
                                            int score,
                                            arma::vec& labels) {
  // check classifications
  int correct = 0;
  for (int i = 0; i < (neighbors.n_elem / knns); i++) {
    arma::vec classifieds(score);
    for (int j = 0; j < score; j++)
      classifieds[j] = labels[neighbors[i * knns + j]];

    int classvote = ClassifyKNN(classifieds);
    int label = (int) labels[i];
    
    if(classvote == label)
      correct++;
  }

  return correct;
}

/***
 * Given a list of calculated neighbors and their associated labels, evaluate
 * the accuracy of the calculated neighbors.  The vector of neighbors should be
 * of the same format as the output of the AllkNN class.
 *
 * @param neighbors List of calculated neighbors (length n_points * knns)
 * @param knns k used in AllkNN calculation
 * @param score Number of neighbors to use for scoring ( <= knns )
 * @param labels List of correct classes of points (length n_points)
 * @return 1 if correctly classified, 0 otherwise.
 */
int mlpack::allknn::metric::EvaluateCorrect(arma::Col<index_t>& neighbors,
                                            int knns,
                                            int score,
                                            arma::vec& labels,
                                            int index) {
  arma::vec classifieds(score);
  for(int j = 0; j < score; j++)
    classifieds[j] = labels[neighbors[j]];
  int classvote = ClassifyKNN(classifieds);
  if(classvote == labels[index])
    return 1;
  
  return 0;
}

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
int mlpack::allknn::metric::EvaluateClassCorrect(
      const arma::Col<index_t>& neighbors,
      int knns,
      const arma::vec& ref_labels,
      const arma::vec& query_labels,
      arma::Col<int>& class_counts,
      arma::Col<int>& class_scores) {

  int num_classes = std::max(max(ref_labels), max(query_labels));
  class_counts.zeros(num_classes);
  class_scores.zeros(num_classes);

  // check classifications
  int correct = 0; // total counter
  
  for (int i = 0; i < (neighbors.n_elem / knns); i++) {
    arma::vec classifieds(knns);
    for (int j = 0; j < knns; j++)
      classifieds[j] = ref_labels[neighbors[i * knns + j]];

    int classvote = ClassifyKNN(classifieds);
    int index = query_labels[i];
    if (index >= num_classes || index < 0)
      FATAL("Bizarre index %d (i %d, label %lf)", index, i, query_labels[i]);

    if(classvote == index) {
      correct++;
      class_scores[index] += 1;
    }

    class_counts[index] += 1;
  }

  return correct;
}
