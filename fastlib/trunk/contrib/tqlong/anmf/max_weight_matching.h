#ifndef MAX_WEIGHT_MATCHING_H
#define MAX_WEIGHT_MATCHING_H

#include "anmf.h"
#include <vector>
#include <limits>
#include <queue>
#include <map>

BEGIN_ANMF_NAMESPACE;

/** Find max weight matching in a weighted bipartie
  * Input: a "matrix" (template) for weights (needs n_rows(), n_cols(), get(i,j) )
  * Output: a max weight matching
  */
template< typename W >
class MaxWeightMatching
{
public:
  typedef W                    weight_matrix_type;
public:
  /** Calculate maximum weight matching of bipartie graph
    * The weight_matrix_type should implement n_rows(), n_cols(), get(i, j) - the weight of (i,j)
    */
  MaxWeightMatching(const weight_matrix_type& M);

  /** The matched vertex on the right vertex set */
  int matchLeft(int u) const;

  /** The matched vertex on the left vertex set */
  int matchRight(int v) const;

  /** The matched pairs as vector of int from left */
  const std::vector<int>& leftMatch() const;

  /** The matched pairs as vector of int from right */
  const std::vector<int>& rightMatch() const;
private:
  /** Init the label such that label[i]+label[j] >= weight(i,j) */
  void initLabel();

  /** Find an alternating path to augment the match */
  int findAugmentingPath(int u);

  /** Augment the match using augmenting path from u to v */
  void augmentPath(int u, int v);

  /** Populate the match pairs */
  void populateMatch();

  /** Calculate weight from the weight matrix */
  double weight(int u, int v) const;

  /** Comparing two doubles using 1e-12 threshold */
  bool equal(double a, double b) const;
private:
  const weight_matrix_type&    M_;
  int                          n_row_, n_col_;
  int                          n_;
  std::vector<int>             match_, prev_;
  std::vector<double>          label_, slack_;
  std::vector<int>             leftMatch_, rightMatch_;
};

template<typename W>
MaxWeightMatching<W>::MaxWeightMatching(const weight_matrix_type &M)
  : M_(M), n_row_(M_.n_rows()), n_col_(M_.n_cols()),
    n_(n_row_ >= n_col_ ? n_row_ : n_col_),
    match_(n_<<1), prev_(n_<<1),
    label_(n_<<1), slack_(n_<<1),
    leftMatch_(n_row_), rightMatch_(n_col_)
{
  initLabel();
  for (int i = 0; i < n_; i++)
  {
    int u = 0;
    while (match_[u] != -1) u++;  // find the next non-matched vertex;
    int v = findAugmentingPath(u);
    augmentPath(u, v);
  }
  populateMatch();
}

template<typename W>
void MaxWeightMatching<W>::initLabel()
{
  for (int i = 0; i < n_; i++)
    label_[i] = 0;
  for (int j = n_; j < n_<<1; j++)
  {
    label_[j] = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_; i++)
      label_[j] = (label_[j] < weight(i, j)) ? weight(i,j) : label_[j];
  }
  for (int j = 0; j < n_<<1; j++)
    match_[j] = -1;
}

// find an augmenting path from a free left vertex
// to a free right vertex in the equality bipartie graph
// (i,j) \in E iff label[i] + label[j] == weight[i,j]
// if no path is found, change the labels to include more edges
template<typename W>
int MaxWeightMatching<W>::findAugmentingPath(int u)
{
  std::queue<int> vertexQueue;
  vertexQueue.push(u);
  for (int i = 0; i < n_<<1; i++) prev_[i] = -1;
  prev_[u] = u;
  // initialize the slacks
  for (int i = n_; i < n_<<1; i++)
    slack_[i] = label_[i]+label_[u]-weight(u, i);
  for (;;)
  {
    DEBUG_ASSERT(!vertexQueue.empty());
    int i = vertexQueue.front();
    // cout << "i = " << i << endl;
    vertexQueue.pop();
    if (i < n_)
    {
      for (int j = n_; j < n_<<1; j++)
      {
        // cout << "j = " << j << " s = " << label_[i]+label_[j]-weight(i,j) << endl;
        if (equal(label_[i]+label_[j], weight(i,j)) && prev_[j] == -1) // if there is an edge (i,j) in the equality graph
        {                                                            // and j is not marked (visited)
          // cout << "j = " << j << endl;
          vertexQueue.push(j);
          prev_[j] = i;
          if (match_[j] == -1) return j; // found a free right vertex
        }
      }
    }
    else // i is a right vertex then i should be matched (not free)
    {
      int j = match_[i];
      if (prev_[j] == -1)
      {
        // cout << "j = " << j << endl;
        vertexQueue.push(j);
        prev_[j] = i;
        // update slacks when new left vertex is visited
        for (int k = n_; k < n_<<1; k++)
          if (prev_[k] == -1 && slack_[k] > label_[k] + label_[j] - weight(j, k))
            slack_[k] = label_[k] + label_[j] - weight(j, k);
      }
    }
    if (!vertexQueue.empty()) continue;
    // if the queue is empty (i.e. no path found under current equality graph)
    // change the labels to include more edges
    // cout << "Change labels" << endl;
    // ot::Print(slack_, "slack before");
    double minSlack = std::numeric_limits<double>::infinity();
    for (int i = n_; i < n_<<1; i++)
      if (prev_[i] == -1 && minSlack > slack_[i]) minSlack = slack_[i];
    // cout << "min slack = " << minSlack << endl;
    DEBUG_ASSERT(minSlack > 0);
    for (int i = 0; i < n_; i++)
      if (prev_[i] != -1) label_[i] -= minSlack;
    for (int i = n_; i < n_<<1; i++)
    {
      if (prev_[i] != -1)
        label_[i] += minSlack;
      else
        slack_[i] -= minSlack;
    }
    // push visited vertices back to the queue
    for (int i = 0; i < n_<<1; i++)
      if (prev_[i] != -1) vertexQueue.push(i);
    // ot::Print(label_, "label");
    // ot::Print(slack_, "slack after");
  }
  return -1; // failure
}

template<typename W>
void MaxWeightMatching<W>::augmentPath(int u, int v)
{
  DEBUG_ASSERT(v >= n_);
  // cout << "augmenting ... ";
  while (v != u)
  {
    int i = prev_[v];
    int j = prev_[i];
    match_[i] = v;
    match_[v] = i;
    v = j;
  }
  // cout << "done" << endl;
}

template<typename W>
void MaxWeightMatching<W>::populateMatch()
{
  for (int i = 0; i < n_; i++)
  {
    int j = match_[i]-n_;
    if (i < n_row_) leftMatch_[i] = j < n_col_ ? j : -1;
    if (j < n_col_) rightMatch_[j] = i < n_row_ ? i : -1;
  }
}

template<typename W>
double MaxWeightMatching<W>::weight(int u, int v) const
{
  if (u > v) { int tmp = u; u = v; v = tmp; }
  v -= n_;
  return u >= n_row_ ? 0 : M_.get(u,v);
}

template<typename W>
bool MaxWeightMatching<W>::equal(double a, double b) const
{
  static const double tol = 1e-12;
  return a-b < tol && b-a < tol;
}

template<typename W>
int MaxWeightMatching<W>::matchLeft(int u) const
{
  return (u < n_row_) ? leftMatch_[u] : -1;
}

template<typename W>
int MaxWeightMatching<W>::matchRight(int v) const
{
  return (v < n_col_) ? rightMatch_[v] : -1;
}

template<typename W>
const std::vector<int>& MaxWeightMatching<W>::leftMatch() const
{
  return leftMatch_;
}

template<typename W>
const std::vector<int>& MaxWeightMatching<W>::rightMatch() const
{
  return rightMatch_;
}

END_ANMF_NAMESPACE;

#endif // MAX_WEIGHT_MATCHING_H
