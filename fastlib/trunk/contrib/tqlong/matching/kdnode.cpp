#include <boost/foreach.hpp>
#include <sstream>
#include "kdnode.h"

MATCHING_NAMESPACE_BEGIN;

KDNode::KDNode(const Matrix &points)
  : points_(points), n_dim_(points.n_rows()), n_points_(points.n_cols()), dfsIndex_(0),
    oldIndex_(*(new index_type(n_points_))), pointToLeaf_(*(new node_list_type(n_points_))),
    parent_(NULL)
{
  for (int i = 0; i < n_points_; i++)
  {
    oldIndex_[i] = i;
  }
}

KDNode::KDNode(KDNode *parent)
  : points_(parent->points_), n_dim_(parent->n_dim_),
    oldIndex_(parent->oldIndex_), pointToLeaf_(parent->pointToLeaf_), parent_(parent)
{
  parent->children_.push_back(this);
}

int KDNode::oldIndex(int index) const
{
  if (index < 0 || index >= n_points_) return -1; // error
  return oldIndex_.at(index+dfsIndex_);
}

void KDNode::getPoint(int index, Vector &point) const
{
  points_.MakeColumnVector(oldIndex(index), &point);
}

double KDNode::get(int dim, int index) const
{
  return points_.get(dim, oldIndex(index));
}

int KDNode::n_points() const
{
  return n_points_;
}

int KDNode::n_dim() const
{
  return n_dim_;
}

int KDNode::n_children() const
{
  return (int) children_.size();
}

KDNode* KDNode::child(int index) const
{
  return children_.at(index);
}

KDNode* KDNode::parent() const
{
  return parent_;
}

bool KDNode::isLeaf() const
{
  return children_.size() == 0;
}

bool KDNode::isRoot() const
{
  return parent_ == NULL;
}

KDNode* KDNode::leaf(int index) const
{
  return pointToLeaf_.at(oldIndex(index));
}

void KDNode::split(int minSize)
{
  if (n_points_ <= minSize) // the node is too small to split
  {
    for (int i = 0; i < n_points_; i++) pointToLeaf_[oldIndex(i)] = this;
  }
  else
  {
    int dim = math::RandInt(0, n_dim_);   // try splitting on a random dimension
    splitOnDim(dim);
    if (children_.size() > 0)
    {
      BOOST_FOREACH(KDNode* child, children_)
      {
        child->split();                     // split the children
      }
    }
    else
    {
      for (int i = 0; i < n_points_; i++) pointToLeaf_[oldIndex(i)] = this;
    }
  }
}

void KDNode::splitOnDim(int dim)
{
  for (int tries = 0; tries < n_dim_; tries++)
  {
    split_.splitDim_ = dim;
    split_.splitVal_ = selectMedian(dim);  // select a good pivot to split
    if (splitNode()) break;                // split the node, if successful, exit
    dim = (dim+1) % n_dim_;
  }
}

bool KDNode::splitNode()
{
  int n_left = 0, n_right = 0;
  for (int i = 0; i < n_points_; i++)
  {
    if (get(split_.splitDim_, i) < split_.splitVal_) n_left++; // calculate the number of points
    else n_right++;                                            // in left and right nodes
  }
  if (n_left == 0 || n_right == 0) return false;

  KDNode *left = newNode(this), *right = newNode(this);
  left->n_points_ = n_left; left->dfsIndex_ = dfsIndex_;
  right->n_points_ = n_right; right->dfsIndex_ = dfsIndex_+n_left;

  index_type oldMap(n_points_);
  int i_left = 0, i_right = n_left;
  for (int i = 0; i < n_points_; i++)                          // adjust the oldIndex_ map
  {                                                            // to reflect the children nodes' points
    if (get(split_.splitDim_, i) < split_.splitVal_) oldMap[i_left++] = oldIndex(i);
    else oldMap[i_right++] = oldIndex(i);
  }
  for (int i = 0; i < n_points_; i++)
    oldIndex_[i+dfsIndex_] = oldMap[i];
  return true;
}

KDNode* KDNode::newNode(KDNode *parent)
{
  return new KDNode(parent);
}

/** Select median of a short array (n <= 5) */
double selectMedianSmall(const std::vector<double>& x, int s, int n)
{
  DEBUG_ASSERT(n != 0);
  double mid;
  if (n == 1) mid = x[s];
  else if (n == 2) mid =  (x[s]+x[s+1])/2;
  else if (n == 3)
  {
    if (x[s] < x[s+1])
    {
      if (x[s+1] < x[s+2]) mid =  x[s+1];
      else if (x[s] < x[s+2]) mid = x[s+2];
      else mid = x[s];
    }
    else
    {
      if (x[s+1] > x[s+2]) mid = x[s+1];
      else if (x[s] > x[s+2]) mid = x[s+2];
      else mid = x[s];
    }
  }
  else if (n == 4)
  {
    double min1, min2;
    double max1, max2;
    min1 = min2 = std::numeric_limits<double>::infinity();
    max1 = max2 = -min1;
    for (int i = 0; i < n; i++)
    {
      if (x[s+i] < min1) { min2 = min1; min1 = x[s+i]; }
      else if (x[s+i] < min2) min2 = x[s+i];
      if (x[s+i] > max1) { max2 = max1; max1 = x[s+i]; }
      else if (x[s+i] > max2) max2 = x[s+i];
    }
    mid = (min2+max2)/2;
  }
  else
  {
    double min1, min2;
    double max1, max2;
    min1 = min2 = std::numeric_limits<double>::infinity();
    max1 = max2 = -min1;
    for (int i = 0; i < n; i++)
    {
      if (x[s+i] < min1) { min2 = min1; min1 = x[s+i]; }
      else if (x[s+i] < min2) min2 = x[s+i];
      if (x[s+i] > max1) { max2 = max1; max1 = x[s+i]; }
      else if (x[s+i] > max2) max2 = x[s+i];
    }
    mid = (min2+max2)/2;
    for (int i = 0; i < n; i++)
      if (x[s+i] < max2 && x[s+i] > min2) mid = x[s+i];
  }
//    std::cout << "x = ";
//    for (int i = 0; i < n; i++)
//      std::cout << x[s+i] << " ";
//    std::cout << "\nmid = " << mid << " n = " << n << "\n";
  return mid;
}

double selectMedian(std::vector<double>& x, int n)
{
  if (n < 5) return selectMedianSmall(x, 0, n);
  int k = 0;
  for (int i = 0; i < n; i+=5)
    x[k++] = selectMedianSmall(x, i, n-i > 5 ? 5 : n-i);
  return selectMedian(x, k);
}

double KDNode::selectMedian(int dim) const
{
  std::vector<double> x(n_points_);
  for (int i = 0; i < n_points_; i++)
    x[i] = get(dim, i);
  return match::selectMedian(x, n_points_);
}

/** convert this subtree to string (to print) */
std::string KDNode::toString(int depth) const
{
  std::stringstream s;
  if (depth == 0)
  {
    for (int i = 0; i < n_points_; i++)
    {
      Vector p_i;
      getPoint(i, p_i);
      s << i << " --> " << match::toString(p_i) << "\n";
    }
  }
  for (int i = 0; i < depth; i++) s << "  ";
  s << "-- Node (" << dfsIndex_;
  for (int i = 1; i < n_points_; i++)
    s << "," << dfsIndex_ + i;
  s << ")\n";
//  for (int i = 0; i < depth+1; i++) s << "  ";
//  s << " " << nodeStatistics_.toString() << "\n";
  for (unsigned int i = 0; i < children_.size(); i++)
    s << children_[i]->toString(depth+1);
  return s.str();
}

MATCHING_NAMESPACE_END;
