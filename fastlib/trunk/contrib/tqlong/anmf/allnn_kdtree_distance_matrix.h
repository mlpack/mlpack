#ifndef KDTREE_DISTANCE_MATRIX_H
#define KDTREE_DISTANCE_MATRIX_H

#include "anmf.h"
#include <limits>
#include <string>
#include <sstream>
#include <boost/foreach.hpp>

BEGIN_ANMF_NAMESPACE;

class PointStatistics
{
  friend class NodeStatistics;

  // main statistics
  double price_;
  int matchTo_;

  // temporaries
  int first_match_, second_match_;
  double first_min_, second_min_;
public:
  PointStatistics(double price = 0.0, int matchTo = -1) : price_(price), matchTo_(matchTo) {}
//  PointStatistics& operator= (double price) { price_ = price; return *this; }
  double price() const { return price_; }
  int matchTo() const { return matchTo_; }
  void getResult(int& first_match, double& first_min, int& second_match, double& second_min) const
  {
    first_match = first_match_;
    first_min = first_min_;
    second_match = second_match_;
    second_min = second_min_;
  }
  void setResult(int first_match, double first_min, int second_match, double second_min)
  {
    first_match_ = first_match;
    first_min_ = first_min;
    second_match_ = second_match;
    second_min_ = second_min;
  }

  double processNewMatch(int match, double val)
  {
    if (val < first_min_)
    {
      second_min_ = first_min_;
      second_match_ = first_match_;
      first_min_ = val;
      first_match_ = match;
    }
    else if (val < second_min_)
    {
      second_min_ = val;
      second_match_ = match;
    }
    return second_min_;
  }
};

class NodeStatistics
{
  // main statistics
  double minPrice_, maxPrice_;
  int n_matches_;
  Vector minBox_, maxBox_;

  // temporaries
  double min_upper_bound_;
public:
  double minPrice() const { return minPrice_; }
  double maxPrice() const { return maxPrice_; }
  const Vector& minBox() const { return minBox_; }
  const Vector& maxBox() const { return maxBox_; }
  int n_matches() const { return n_matches_; }
  double minUpperBound() const { return min_upper_bound_; }
  void setMinUpperBound(double min_upper_bound) { min_upper_bound_ = min_upper_bound; }

  double distance(const Vector& x) const
  {
    double s = 0;
    for (int i = 0; i < minBox_.length(); i++)
    {
      if (x[i] < minBox_[i]) s += math::Sqr(x[i]-minBox_[i]);
      else if (x[i] > maxBox_[i]) s +=  math::Sqr(x[i]-maxBox_[i]);
    }
    return sqrt(s);
  }

  double distance(const NodeStatistics& stats) const
  {
    double s = 0;
    for (int i = 0; i < minBox_.length(); i++)
    {
      if (maxBox_[i] < stats.minBox_[i]) s += math::Sqr(maxBox_[i]-stats.minBox_[i]);
      else if (stats.maxBox_[i] < minBox_[i]) s += math::Sqr(stats.maxBox_[i]-minBox_[i]);
    }
    return s;
  }

  /** Create bounding box for prices and coordinates of leaf node */
  void InitAtLeaf(const Matrix& points, const std::vector<PointStatistics>& pointStatistics,
                  const std::vector<int>& indexMap, int dfsIndex, int n_points)
  {
    minPrice_ = std::numeric_limits<double>::infinity();
    maxPrice_ = -std::numeric_limits<double>::infinity();
    minBox_.Init(points.n_rows());
    maxBox_.Init(points.n_rows());
    minBox_.SetAll(std::numeric_limits<double>::infinity());
    maxBox_.SetAll(-std::numeric_limits<double>::infinity());
    n_matches_ = n_points;
    for (int i = 0; i < n_points; i++)
    {
      int index = dfsIndex+i;
      int oldIndex = indexMap[index];

      double price = pointStatistics[oldIndex].price_;
      if (pointStatistics[oldIndex].matchTo_ == -1) n_matches_--;
      if (price < minPrice_) minPrice_ = price;
      if (price > maxPrice_) maxPrice_ = price;

      for (int dim = 0; dim < points.n_rows(); dim++)
      {
        double val = points.get(dim, oldIndex);
        if (val < minBox_[dim]) minBox_[dim] = val;
        if (val > maxBox_[dim]) maxBox_[dim] = val;
      }
    }
  }

  void InitFromChildren(const NodeStatistics& leftStats, const NodeStatistics& rightStats)
  {
    minBox_.Init(leftStats.minBox_.length());
    maxBox_.Init(leftStats.maxBox_.length());
    minPrice_ = leftStats.minPrice_ < rightStats.minPrice_ ? leftStats.minPrice_ : rightStats.minPrice_;
    maxPrice_ = leftStats.maxPrice_ > rightStats.maxPrice_ ? leftStats.maxPrice_ : rightStats.maxPrice_;
    n_matches_ = leftStats.n_matches_ + rightStats.n_matches_;
    for (int dim = 0; dim < minBox_.length(); dim++)
    {
      minBox_[dim] = leftStats.minBox_[dim] < rightStats.minBox_[dim] ? leftStats.minBox_[dim] : rightStats.minBox_[dim];
      maxBox_[dim] = leftStats.maxBox_[dim] > rightStats.maxBox_[dim] ? leftStats.maxBox_[dim] : rightStats.maxBox_[dim];
    }
  }

  void ResetAtLeaf(const Matrix& points, const std::vector<PointStatistics>& pointStatistics,
                   const std::vector<int>& indexMap, int dfsIndex, int n_points, bool resetBoundingBox = false)
  {
    minPrice_ = std::numeric_limits<double>::infinity();
    maxPrice_ = -std::numeric_limits<double>::infinity();
    if (resetBoundingBox)
    {
      minBox_.SetAll(std::numeric_limits<double>::infinity());
      maxBox_.SetAll(-std::numeric_limits<double>::infinity());
    }

    n_matches_ = n_points;
    for (int i = 0; i < n_points; i++)
    {
      int index = dfsIndex+i;
      int oldIndex = indexMap[index];

      double price = pointStatistics[oldIndex].price_;
      if (pointStatistics[oldIndex].matchTo_ == -1) n_matches_--;
      if (price < minPrice_) minPrice_ = price;
      if (price > maxPrice_) maxPrice_ = price;

      if (resetBoundingBox)
      {
        for (int dim = 0; dim < points.n_rows(); dim++)
        {
          double val = points.get(dim, oldIndex);
          if (val < minBox_[dim]) minBox_[dim] = val;
          if (val > maxBox_[dim]) maxBox_[dim] = val;
        }
      }
    }
  }

  void ResetFromChildren(const NodeStatistics& leftStats, const NodeStatistics& rightStats, bool resetBoundingBox = false)
  {
    minPrice_ = leftStats.minPrice_ < rightStats.minPrice_ ? leftStats.minPrice_ : rightStats.minPrice_;
    maxPrice_ = leftStats.maxPrice_ > rightStats.maxPrice_ ? leftStats.maxPrice_ : rightStats.maxPrice_;
    n_matches_ = leftStats.n_matches_ + rightStats.n_matches_;
    if (resetBoundingBox)
      for (int dim = 0; dim < minBox_.length(); dim++)
      {
        minBox_[dim] = leftStats.minBox_[dim] < rightStats.minBox_[dim] ? leftStats.minBox_[dim] : rightStats.minBox_[dim];
        maxBox_[dim] = leftStats.maxBox_[dim] > rightStats.maxBox_[dim] ? leftStats.maxBox_[dim] : rightStats.maxBox_[dim];
      }
  }

  std::string toString() const
  {
    std::stringstream s;
    s << "price = (" << minPrice_ << "," << maxPrice_ << ")"
        << " box = " << anmf::toString(minBox_) << " --> " << anmf::toString(maxBox_);
    return s.str();
  }
};

class KDNode
{
protected:
  /** Global properties of a tree */
  const Matrix& points_;
  std::vector<PointStatistics>* pointStatistics_;
  std::vector<int>* oldFromNewIndex_;
  std::vector<KDNode*> *nodeFromOldIndex_;

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
    nodeFromOldIndex_ = new std::vector<KDNode*>(n);
    for (int i = 0; i < n; i++)
    {
      oldFromNewIndex_->at(i) = i;
      pointStatistics_->at(i) = stats[i];
      nodeFromOldIndex_->at(i) = NULL;
    }
    n_points_ = n;
    dfsIndex_ = 0;
    parent_ = NULL;
    splitMidPoint(0);
    visitToSetStatistics();
  }

  /** set point statistics and traverse up the tree */
  void setPointStatistics(int index, const PointStatistics& stats)
  {
    pointStats(index) = stats;
    leaf(index)->resetStatistics();
  }

  /** convert this subtree to string (to print) */
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
        s << anmf::toString(p_i) << "\n";
      }
    }
    for (int i = 0; i < depth; i++) s << "  ";
    s << "-- Node (" << dfsIndex_;
    for (int i = 1; i < n_points_; i++)
      s << "," << dfsIndex_ + i;
    s << ")\n";
    for (int i = 0; i < depth+1; i++) s << "  ";
    s << " " << nodeStatistics_.toString() << "\n";
    for (unsigned int i = 0; i < children_.size(); i++)
      s << children_[i]->toString(depth+1);
    return s.str();
  }

  /** choose a random point to set the bounds */
  void randomBound(const Vector& x, int& minIndex, double& minSoFar)
  {
    minIndex = dfsIndex_ + math::RandInt(0, n_points_);
//    minIndex = dfsIndex_ + math::RandInt(0, n_points_);
    int oldIndex = oldFromNewIndex(minIndex);
    Vector p_i;
    points_.MakeColumnVector(oldIndex, &p_i);
    double d = sqrt(la::DistanceSqEuclidean(p_i, x));
    double p = pointStats(minIndex).price();
    minSoFar = d+p;
  }

  void randomBound(const Vector& x, int& minIndex, double& minSoFar, int& sndIndex, double& sndMinSoFar)
  {
    minIndex = dfsIndex_ + math::RandInt(0, n_points_);
//    minIndex = dfsIndex_ + math::RandInt(0, n_points_);
    int oldIndex = oldFromNewIndex(minIndex);
    Vector p_i;
    points_.MakeColumnVector(oldIndex, &p_i);
    double d = sqrt(la::DistanceSqEuclidean(p_i, x));
    double p = pointStats(minIndex).price();
    minSoFar = d+p;

    sndIndex = dfsIndex_ + math::RandInt(0, n_points_);
//    minIndex = dfsIndex_ + math::RandInt(0, n_points_);
    oldIndex = oldFromNewIndex(sndIndex);
    Vector p_j;
    points_.MakeColumnVector(oldIndex, &p_j);
    d = sqrt(la::DistanceSqEuclidean(p_j, x));
    p = pointStats(sndIndex).price();
    sndMinSoFar = d+p;

    if (minSoFar > sndMinSoFar)
    {
      int tmp = minIndex; minIndex = sndIndex; sndIndex = tmp;
      double dtmp = minSoFar; minSoFar = sndMinSoFar; sndMinSoFar = dtmp;
    }
  }
  /** get Nearest Neighbor index (newIndex) in term of distance + price */
  void nearestNeighbor(const Vector& x, int& minIndex, double& minSoFar)
  {
    // check bound
    double d = nodeStatistics_.distance(x);
    double minPrice = nodeStatistics_.minPrice();
//    std::cout << "lb = " << d+minPrice << " minIndex = " << minIndex << " minSoFar = " << minSoFar << "\n";
    if (d+minPrice >= minSoFar)
    {
//      std::cout << "PRUNE  " << nodeStatistics_.toString() << " n_points = " << n_points_ << "\n";
      return;
    }
    else
    {
//      std::cout << "Search " << nodeStatistics_.toString() << " n_points = " << n_points_ << "\n";
    }

    if (isLeaf()) // at leaf, do naive search
    {
      for (int i = dfsIndex_; i < dfsIndex_+n_points_; i++)
      {
        int oldIndex = oldFromNewIndex(i);
        Vector p_i;
        points_.MakeColumnVector(oldIndex, &p_i);
        double d = sqrt(la::DistanceSqEuclidean(p_i, x));
        double p = pointStats(i).price();
        if (d+p < minSoFar)
        {
          minSoFar = d+p;
          minIndex = i;
        }
      }
    }
    else
    {
      children_[0]->nearestNeighbor(x, minIndex, minSoFar);
      children_[1]->nearestNeighbor(x, minIndex, minSoFar);
    }
  }

  /** return the number of pruned calculations */
  long int nearestNeighbor(const Vector& x, int& minIndex, double& minSoFar, int& sndIndex, double& sndMinSoFar)
  {
    // check bound
    double d = nodeStatistics_.distance(x);
    double minPrice = nodeStatistics_.minPrice();
//    std::cout << "lb = " << d+minPrice << " minIndex = " << minIndex << " minSoFar = " << minSoFar << "\n";
    if (d+minPrice >= sndMinSoFar)
    {
//      std::cout << "PRUNE  " << nodeStatistics_.toString() << " n_points = " << n_points_ << "\n";
      return n_points_;
    }
    else
    {
//      std::cout << "Search " << nodeStatistics_.toString() << " n_points = " << n_points_ << "\n";
    }

    if (isLeaf()) // at leaf, do naive search
    {
      for (int i = dfsIndex_; i < dfsIndex_+n_points_; i++)
      {
        int oldIndex = oldFromNewIndex(i);
        Vector p_i;
        points_.MakeColumnVector(oldIndex, &p_i);
        double d = sqrt(la::DistanceSqEuclidean(p_i, x));
        double p = pointStats(i).price();
        if (d+p < minSoFar)
        {
          sndIndex = minIndex;
          sndMinSoFar = minSoFar;
          minIndex = i;
          minSoFar = d+p;
        }
        else if (d+p < sndMinSoFar)
        {
          sndIndex = i;
          sndMinSoFar = d+p;
        }
      }
      return 0;
    }
    else
    {
      long int left_pruned = children_[0]->nearestNeighbor(x, minIndex, minSoFar, sndIndex, sndMinSoFar);
      long int right_pruned = children_[1]->nearestNeighbor(x, minIndex, minSoFar, sndIndex, sndMinSoFar);
      return left_pruned + right_pruned;
    }
  }

  /** clear temporaries before dual-tree search */
  void clearTemporaries()
  {
    if (isLeaf())
    {
      nodeStatistics_.setMinUpperBound(std::numeric_limits<double>::infinity());
      for (int i = dfsIndex_; i < dfsIndex_+n_points_; i++)
        pointStats(i).setResult(-1, std::numeric_limits<double>::infinity(), -1,std::numeric_limits<double>::infinity());
    }
    else
    {
      child(0)->clearTemporaries();
      child(1)->clearTemporaries();
    }
  }

  /** Find nearest neighbors of R's points in Q */
  static long int DualTreeNearestNeighbor(KDNode* R, KDNode* Q)
  {
    double min_upper_bound_ = R->nodeStatistics_.minUpperBound();
    if (R->allMatched())
      return 0;
    if (R->nodeStatistics_.distance(Q->nodeStatistics_) >= min_upper_bound_)
      return (long int) (R->n_points()-R->stats().n_matches())* (long int)Q->n_points();

    if (R->isLeaf() and Q->isLeaf())
      return DualTreeNearestNeighborBase(R, Q);
    else if (R->isLeaf() and !Q->isLeaf())
    {
      long int pruned = DualTreeNearestNeighbor(R, Q->child(0));
      pruned += DualTreeNearestNeighbor(R, Q->child(1));
      return pruned;
    }
    else if (!R->isLeaf() and Q->isLeaf())
    {
      long int pruned = DualTreeNearestNeighbor(R->child(0), Q);
      pruned += DualTreeNearestNeighbor(R->child(1), Q);
      return pruned;
    }
    else
    {
      long int pruned = DualTreeNearestNeighbor(R->child(0), Q->child(0));
      pruned += DualTreeNearestNeighbor(R->child(1), Q->child(0));
      pruned += DualTreeNearestNeighbor(R->child(0), Q->child(1));
      pruned += DualTreeNearestNeighbor(R->child(1), Q->child(1));
      return pruned;
    }
  }

  static long int DualTreeNearestNeighborBase(KDNode* R, KDNode* Q)
  {
    DEBUG_ASSERT(R->isLeaf() && Q->isLeaf());

    double min_upper_bound = -std::numeric_limits<double>::infinity();
    for (int i = R->dfsIndex_; i < R->dfsIndex_+R->n_points_; i++) if (R->pointStats(i).matchTo() == -1)
    {
      Vector r_i;
      R->getPoint(i, r_i);
      for (int j = Q->dfsIndex_; j < Q->dfsIndex_+Q->n_points_; j++)
      {
        Vector q_j;
        Q->getPoint(j, q_j);
        double bound = R->pointStats(i).processNewMatch(j, sqrt(la::DistanceSqEuclidean(r_i, q_j))+Q->pointStats(j).price());
        if (min_upper_bound < bound) min_upper_bound = bound;
      }
    }

    R->setMinUpperBound(min_upper_bound);

    return 0;
  }

  /** Basic getters and setters */
  void getPoint(int index, Vector& p) { points_.MakeColumnVector(oldFromNewIndex(index), &p); }
  int n_points() const { return n_points_; }
  int dfsIndex() const { return dfsIndex_; }
//  NodeStatistics& stats() { return nodeStatistics_; }
  const NodeStatistics& stats() const { return nodeStatistics_; }
  KDNode* parent() const { return parent_; }
  int n_children() const { return (int) children_.size(); }
  KDNode* child(int index) const { return children_[index]; }
  const PointStatistics& pointStats(int index) const { return pointStatistics_->at(oldFromNewIndex(index)); }
  PointStatistics& pointStats(int index) { return pointStatistics_->at(oldFromNewIndex(index)); }
  int oldFromNewIndex(int index) const { return oldFromNewIndex_->at(index); }
  KDNode* leaf(int index) const { return nodeFromOldIndex_->at(oldFromNewIndex(index)); }
  bool isLeaf() const { return children_.empty(); }
  bool allMatched() const { return nodeStatistics_.n_matches() == n_points_; }
protected:

  /** Constructor for a child node */
  KDNode(KDNode* parent)
    : points_(parent->points_),
      pointStatistics_(parent->pointStatistics_),
      oldFromNewIndex_(parent->oldFromNewIndex_),
      nodeFromOldIndex_(parent->nodeFromOldIndex_),
      parent_(parent)
  {
    DEBUG_ASSERT(parent);
    parent->children_.push_back(this);
  }

  /** Split the points in a node by the median point at certain dimension */
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

  /** find the median point at a certain dimension
    * try other dimensions if cannot split
    */
  double findMidPoint(int& dim, int& n_left, int& n_right)
  {
    std::vector<double> vals(n_points_);
    for (int k = 0; k < points_.n_rows(); k++)
    {
      n_left = n_right = 0;
      for (int i = 0; i < n_points_; i++)
        vals[i] = points_.get(dim, oldFromNewIndex_->at(i+dfsIndex_));

      double mid = selectMedian(vals, n_points_); // using median of medians algorithm here

      // check if the split is ok
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

  /** The median of medians algorithm */
  double selectMedian(std::vector<double>& x, int n)
  {
    if (n < 5) return selectMedianSmall(x, 0, n);
    int k = 0;
    for (int i = 0; i < n; i+=5)
      x[k++] = selectMedianSmall(x, i, n-i > 5 ? 5 : n-i);
    return selectMedian(x, k);
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

  /** Traverse tree to set node statistics */
  void visitToSetStatistics()
  {
    if (children_.size() > 0)
    {
      BOOST_FOREACH(KDNode* child, children_)
      {
        child->visitToSetStatistics();
      }
      nodeStatistics_.InitFromChildren(children_[0]->nodeStatistics_, children_[1]->nodeStatistics_);
    }
    else
    {
      nodeStatistics_.InitAtLeaf(points_, *pointStatistics_, *oldFromNewIndex_, dfsIndex_, n_points_);
      for (int i = dfsIndex_; i < dfsIndex_+n_points_; i++)
        nodeFromOldIndex_->at(oldFromNewIndex(i)) = this;
    }
  }

  /** Traverse up the tree reset node statistics after a change at the leaves */
  void resetStatistics()
  {
    if (children_.empty())
    {
      nodeStatistics_.ResetAtLeaf(points_, *pointStatistics_, *oldFromNewIndex_, dfsIndex_, n_points_);
    }
    else
    {
      nodeStatistics_.ResetFromChildren(children_[0]->nodeStatistics_, children_[1]->nodeStatistics_);
    }
    if (parent_)
      parent_->resetStatistics();
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

  void setMinUpperBound(double min_upper_bound)
  {
    if (isLeaf())
      nodeStatistics_.setMinUpperBound(min_upper_bound);
    else
    {
      double left = child(0)->nodeStatistics_.minUpperBound();
      double right = child(1)->nodeStatistics_.minUpperBound();
      nodeStatistics_.setMinUpperBound(left > right ? left : right);
    }
    if (parent_)
      parent_->setMinUpperBound(min_upper_bound);
  }

};

class KDTreeDistanceMatrix
{
  const Matrix &reference_, &query_;
  std::vector<double> price_;
  KDNode *referenceRoot_, *queryRoot_;
public:
  KDTreeDistanceMatrix(const Matrix &reference, const Matrix &query)
    : reference_(reference), query_(query), price_(query.n_cols(), 0)
  {
    DEBUG_ASSERT(reference.n_rows() == query.n_rows());
    referenceRoot_ = new KDNode(reference_, price_);
    queryRoot_ = new KDNode(query_, price_);
  }
  int n_rows() const { return reference_.n_cols(); }
  int n_cols() const { return query_.n_cols(); }
  void row(int i, Vector& v) { referenceRoot_->getPoint(i, v); }
  void col(int j, Vector& v) { queryRoot_->getPoint(j, v); }
  double get(int i, int j) const
  {
    Vector r_i, q_j;
    reference_.MakeColumnVector(referenceRoot_->oldFromNewIndex(i), &r_i);
    query_.MakeColumnVector(queryRoot_->oldFromNewIndex(j), &q_j);
    return -sqrt(la::DistanceSqEuclidean(r_i, q_j));
  }
  void setPrice(int j, const PointStatistics& price)
  {
    PointStatistics pStats = queryRoot_->pointStats(j);
    DEBUG_ASSERT(pStats.matchTo() != price.matchTo());
    queryRoot_->setPointStatistics(j, price);
    if (pStats.matchTo() != -1)
      referenceRoot_->setPointStatistics(pStats.matchTo(), PointStatistics(0, -1));
    referenceRoot_->setPointStatistics(price.matchTo(), PointStatistics(0, j));
  }
  long int getBestAndSecondBest(int bidder, int &best_item, double &best_surplus, double &second_surplus)
  {
    Vector r_i;
    reference_.MakeColumnVector(referenceRoot_->oldFromNewIndex(bidder), &r_i);

    int minIndex, sndIndex;
    double min, sndMin;
    queryRoot_->randomBound(r_i, minIndex, min, sndIndex, sndMin);
    int pruned = queryRoot_->nearestNeighbor(r_i, minIndex, min, sndIndex, sndMin);
    best_item = minIndex;
    best_surplus = -min;
    second_surplus = -sndMin;
//    std::cout << "bidder = " << bidder << " best = " << best_item
//        << " best surplus = " << best_surplus << " second surplus = " << second_surplus << "\n";
    return pruned;
  }
  bool allMatched() const
  {
    return referenceRoot_->allMatched();
  }

  long int getAllBestAndSecondBest(std::vector<int>& best_item, std::vector<double>& best_surplus, std::vector<double>& second_surplus)
  {
//    int pruned = 0;
//    for (int bidder = 0; bidder < n_rows(); bidder++) if (referenceRoot_->pointStats(bidder).matchTo() == -1)
//    {
//      pruned += getBestAndSecondBest(bidder, best_item[bidder], best_surplus[bidder], second_surplus[bidder]);
//    }
    referenceRoot_->clearTemporaries();
    long int pruned = KDNode::DualTreeNearestNeighbor(referenceRoot_, queryRoot_);
    for (int i = 0; i < n_rows(); i++) if (referenceRoot_->pointStats(i).matchTo() == -1)
    {
      int tmp;
      referenceRoot_->pointStats(i).getResult(best_item[i], best_surplus[i], tmp, second_surplus[i]);
      best_surplus[i] = -best_surplus[i];
      second_surplus[i] = -second_surplus[i];
    }
    return pruned;
  }
};

END_ANMF_NAMESPACE;

#endif // KDTREE_DISTANCE_MATRIX_H
