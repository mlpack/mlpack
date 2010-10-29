#ifndef KDTREE_DISTANCE_MATRIX_H
#define KDTREE_DISTANCE_MATRIX_H

#include "anmf.h"
#include <limits>
#include <string>
#include <sstream>

BEGIN_ANMF_NAMESPACE;

class PointStatistics
{
  double price_;
public:
  PointStatistics(double price = 0.0) : price_(price) {}
  PointStatistics& operator= (double price) { price_ = price; return *this; }
};

class NodeStatistics
{
public:
  void InitAtLeaf(const Matrix& points, const std::vector<PointStatistics>& pointStatistics,
                  const std::vector<int>& indexMap, int dfsIndex, int n_points)
  {

  }

  void InitFromChildren(const NodeStatistics& leftStats, const NodeStatistics& rightStats)
  {

  }
};


std::ostream& operator << (std::ostream& s, const Vector& v)
{
  s << "(" << v[0];
  for (int i = 1; i < v.length(); i++)
    s << "," << v[i];
  s << ")";
  return s;
}

class KDNode
{
protected:
  /** Global properties of a tree */
  const Matrix& points_;
  std::vector<PointStatistics>* pointStatistics_;
  std::vector<int>* oldFromNewIndex_;

  /** Node properties */
  int n_points_, dfsIndex_;
  NodeStatistics nodeStatistics_;
  std::vector<KDNode*> children_;
  KDNode* parent_;
public:
  /** Constructor for root node */
  template <typename W>
      KDNode(const Matrix& points, const std::vector<W>& stats)
    : points_(points)
  {
    //    create global properties for the tree
    int n = points_.n_cols();
    oldFromNewIndex_ = new std::vector<int>(n);
    pointStatistics_ = new std::vector<PointStatistics>(n);
    for (int i = 0; i < n; i++)
    {
      oldFromNewIndex_->at(i) = i;
      pointStatistics_->at(i) = stats[i];
    }
    n_points_ = n;
    dfsIndex_ = 0;
    parent_ = NULL;
    splitMidPoint(0);
  }

  std::string toString(int depth = 0) const
  {
    std::stringstream s;
    if (depth == 0)
    {
      for (int i = 0; i < n_points_; i++)
      {
        int oldIndex = oldFromNewIndex_->at(i);
        s << i << " --> ";
        Vector p_i;
        points_.MakeColumnVector(oldIndex, &p_i);
        s << p_i << "\n";
      }
    }
    for (int i = 0; i < depth; i++) s << "  ";
    s << "-- Node (" << dfsIndex_;
    for (int i = 1; i < n_points_; i++)
      s << "," << dfsIndex_ + i;
    s << ")\n";
    for (unsigned int i = 0; i < children_.size(); i++)
      s << children_[i]->toString(depth+1);
    return s.str();
  }
protected:
  KDNode(KDNode* parent)
    : points_(parent->points_),
      pointStatistics_(parent->pointStatistics_),
      oldFromNewIndex_(parent->oldFromNewIndex_),
      parent_(parent)
  {
    DEBUG_ASSERT(parent);
    parent->children_.push_back(this);
  }

  void splitMidPoint(int dim)
  {
//    std::cout << dfsIndex_ << " --> " << dfsIndex_+n_points_-1 << "\n";
    if (n_points_ < 10) return;  // the node is too small to split

    int n_left = 0, n_right = 0;
    double mid = findMidPoint(dim, n_left, n_right);  // find mid value at dim dimension
    if (n_left == 0 || n_right == 0)  // no mid value found
    {
//      std::cout << "cannot split\n";
      return;
    }

    KDNode *left = new KDNode(this), *right = new KDNode(this);  // create two left and right nodes
    left->n_points_ = n_left;
    left->dfsIndex_ = this->dfsIndex_;
    right->n_points_ = n_right;
    right->dfsIndex_ = this->dfsIndex_ + left->n_points_;

    // now move the points to theirs right location
    n_left = n_right = 0;
    std::vector<int> *oldMap = new std::vector<int>(oldFromNewIndex_->begin()+dfsIndex_, oldFromNewIndex_->begin()+dfsIndex_+n_points_);
    for  (int i = 0; i < n_points_; i++)  // for each point assign to left or right node by its value at dim
    {
      int oldIndex = oldMap->at(i);
      double val = points_.get(dim, oldIndex);
      if (val < mid)
      {
        oldFromNewIndex_->at(left->dfsIndex_+n_left) = oldIndex;
        n_left++;
      }
      else
      {
        oldFromNewIndex_->at(right->dfsIndex_+n_right) = oldIndex;
        n_right++;
      }
    }
    delete oldMap;

    dim = (dim+1) % points_.n_rows();
    left->splitMidPoint(dim);
    right->splitMidPoint(dim);
  }

  double findMidPoint(int& dim, int& n_left, int& n_right)
  {
    std::vector<double> vals(n_points_);
    for (int k = 0; k < points_.n_rows(); k++)
    {
      n_left = n_right = 0;
      for (int i = 0; i < n_points_; i++)
        vals[i] = points_.get(dim, oldFromNewIndex_->at(i+dfsIndex_));

      double mid = selectMedian(vals, n_points_);

      for (int i = 0; i < n_points_; i++)
      {
        int newIndex = i+dfsIndex_;
        int oldIndex = oldFromNewIndex_->at(newIndex);
        double val = points_.get(dim, oldIndex);
//        std::cout << "val = " << val << "\n";
        if (val < mid) n_left++;
        else n_right++;
      }
//      std::cout << "dim = " << dim << " mid = " << mid
//                << " n_left = " << n_left << " n_right = " << n_right << "\n";
      if (n_left >= 1 && n_right >= 1) return mid;
      else
        dim = (dim+1) % points_.n_rows();
    }
    n_left = 0;
    n_right = 0;
    return std::numeric_limits<double>::quiet_NaN();
  }

  double selectMedian(std::vector<double>& x, int n)
  {
    if (n < 5) return selectMedianSmall(x, 0, n);
    int k = 0;
    for (int i = 0; i < n; i+=5)
      x[k++] = selectMedianSmall(x, i, n-i > 5 ? 5 : n-i);
    return selectMedian(x, k);
  }

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

  double findMidPoint1(int& dim, int& n_left, int& n_right)
  {
    for (int k = 0; k < points_.n_rows(); k++)
    {
      n_left = n_right = 0;
      double min = std::numeric_limits<double>::infinity();
      double max = -std::numeric_limits<double>::infinity();
      for (int i = 0; i < n_points_; i++)
      {
        int newIndex = i+dfsIndex_;
        int oldIndex = oldFromNewIndex_->at(newIndex);
        double val = points_.get(dim, oldIndex);
        if (val > max) max = val;
        if (val < min) min = val;
      }
      double mid = (min+max)/2;

      for (int i = 0; i < n_points_; i++)
      {
        int newIndex = i+dfsIndex_;
        int oldIndex = oldFromNewIndex_->at(newIndex);
        double val = points_.get(dim, oldIndex);
        std::cout << "val = " << val << "\n";
        if (val < mid) n_left++;
        else n_right++;
      }

      std::cout << "dim = " << dim << " (min, mid, max) = " << min << ", " << mid << ", " << max
                << " n_left = " << n_left << " n_right = " << n_right << "\n";
      if (n_left >= 1 && n_right >= 1) return mid;
      else
        dim = (dim+1) % points_.n_rows();
    }
    n_left = 0;
    n_right = 0;
    return std::numeric_limits<double>::quiet_NaN();
  }
};

class KDTreeDistanceMatrix
{
  const Matrix &reference_, &query_;
  std::vector<double> price_;
  KDNode* queryRoot_;
public:
  KDTreeDistanceMatrix(const Matrix &reference, const Matrix &query)
    : reference_(reference), query_(query), price_(query.n_cols(), 0)
  {
    DEBUG_ASSERT(reference.n_rows() == query.n_rows());
    queryRoot_ = new KDNode(query_, price_);
  }
  int n_rows() const { return reference_.n_cols(); }
  int n_cols() const { return query_.n_cols(); }
  double get(int i, int j) const
  {
    Vector r_i, q_j;
    reference_.MakeColumnVector(i, &r_i);
    query_.MakeColumnVector(j, &q_j);
    return -sqrt(la::DistanceSqEuclidean(r_i, q_j));
  }
  void setPrice(int j, double price)
  {
    price_[j] = price;
  }
  void getBestAndSecondBest(int bidder, int &best_item, double &best_surplus, double &second_surplus)
  {
    best_surplus = second_surplus = -std::numeric_limits<double>::infinity();
    for (int item = 0; item < query_.n_cols(); item++)
    {
      double surplus = get(bidder, item) - price_[item];
      if (surplus > best_surplus)
      {
        best_item = item;
        second_surplus = best_surplus;
        best_surplus = surplus;
      }
      else if (surplus > second_surplus)
      {
        second_surplus = surplus;
      }
    }
  }
};

END_ANMF_NAMESPACE;

#endif // KDTREE_DISTANCE_MATRIX_H
