/**
 * @file kd.h
 *
 * Pointerless versions of everything needed to make different kinds of
 * trees.
 *
 * Eventually we'll have to figure out how to make these dynamic -- this
 * task ain't no trivial thing.
 */

#ifndef SUPERPAR_KD_H
#define SUPERPAR_KD_H

/**
 * A binary space partitioning tree, such as KD or ball tree, for use
 * with super-par.
 *
 * This particular tree forbids you from having more children.
 *
 * @param TBound the bounding type of each child (TODO explain interface)
 * @param TDataset the data set type
 * @param TStatistic extra data in the node
 *
 * @experimental
 */
template<class TBound,
         int t_cardinality = 2>
class SpNode {
 public:
  typedef TBound Bound;
  typedef TDataset Dataset;
  typedef TStatistic Statistic;
  
  enum {
    /** The root node of a tree is always at index zero. */
    ROOT_INDEX = 0;
  };
  
 private:
  index_t begin_;
  index_t count_;
  
  Bound bound_;

  index_t children_[t_cardinality];
  
 public:
  SpNode() {
    DEBUG_ONLY(begin_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(count_ = BIG_BAD_NUMBER);
    mem::DebugPoison(&children_, t_cardinality);
  }  

  ~SpNode() {
    DEBUG_ONLY(begin_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(count_ = BIG_BAD_NUMBER);
    mem::DebugPoison(&children_, t_cardinality);
  }
  
  void Init(index_t begin_in, index_t count_in) {
    DEBUG_ASSERT(begin_ == BIG_BAD_NUMBER);
    begin_ = begin_in;
    count_ = count_in;
  }
  
  void set_child(index_t i, SpNode *child) {
    DEBUG_BOUNDS(i, t_cardinality);
    children_[i] = child;
  }

  const Bound& bound() const {
    return bound_;
  }

  Bound& bound() {
    return bound_;
  }

  bool is_leaf() const {
    return !left_;
  }

  index_t child(index_t i) const {
    return children_[i];
  }

  /**
   * Gets the index of the first point of this subset.
   */
  index_t begin() const {
    return begin_;
  }

  /**
   * Gets the index one beyond the last index in the series.
   */
  index_t end() const {
    return begin_ + count_;
  }
  
  /**
   * Gets the number of points in this subset.
   */
  index_t count() const {
    return count_;
  }
  
  /**
   * Returns the number of children of this node.
   */
  index_t cardinality() const {
    return t_cardinality;
  }
  
  void PrintSelf() const {
    printf("node: %d to %d: %d points total\n",
       begin_, begin_ + count_ - 1, count_);
  }
};


template<int t_length>
class StaticVector {
 private:
  double array_[t_length];

 public:
  void MakeVector(Vector* alias) {
    alias->Alias(array_, t_length);
  }

  void MakeSubvector(index_t begin, index_t count, Vector *alias) {
    DEBUG_BOUNDS(begin, t_length);
    DEBUG_BOUNDS(begin + count, t_length + 1);
    alias->Alias(array_ + begin, t_length + count);
  }
  
  index_t length() const {
    return t_length;
  }

  double *ptr() {
    return array_;
  }
  
  const double *ptr() const {
    return array_;
  }
  
  double operator [] (index_t i) const {
    DEBUG_BOUNDS(i, t_length);
    return array_[i];
  }
  
  double &operator [] (index_t i) {
    DEBUG_BOUNDS(i, t_length);
    return array_[i];
  }
  
  double get(index_t i) const {
    DEBUG_BOUNDS(i, t_length);
    return array_[i];
  }
};

template<int t_length>
class StaticHrectBound {
 public:
  struct DBound {
    double lo_;
    double hi;
  };
  
 private:
  DBound bounds_[t_length];
  
 public:
  bool Belongs(const Vector& point) const {
    for (index_t i = 0; i < point.length(); i++) {
      const DBound *bound = &bounds_[i];
      if (point[i] > bound->hi || point[i] < bound->lo) {
        return false;
      }
    }
    
    return true;
  }
  
  double MinDistanceSqToInstance(const Vector& point) const {
    DEBUG_ASSERT(point.length() == t_length);
    return MinDistanceSqToInstance(point.ptr());
  }
  
  double MinDistanceSqToInstance(const double *mpoint) const {
    double sumsq = 0;
    const DBound *mbound = bounds_;
    index_t d = t_length;
    
    do {
      double v = *mpoint;
      double v1 = mbound->lo - v;
      double v2 = v - mbound->hi;
      
      v = (v1 + fabs(v1)) + (v2 + fabs(v2));
      
      mbound++;
      mpoint++;
      
      sumsq += v * v;
    } while (--d);

    return sumsq / 4;
  }
  
  double MaxDistanceSqToInstance(const Vector& point) const {
    double sumsq = 0;

    DEBUG_ASSERT(point.length() == t_length);

    for (index_t d = 0; d < t_length; d++) {
      double v = max(point[d] - bounds_[d].lo,
          bounds_[d].hi - point[d]);
      
      sumsq += v * v;
    }

    return sumsq;
  }
  
  double MinDistanceSqToBound(const StaticHrectBound& other) const {
    double sumsq = 0;
    const DBound *a = this->bounds_;
    const DBound *b = other.bounds_;
    index_t mdim = t_length;

    DEBUG_ASSERT(t_length == other.t_length);

    // We invoke the following:
    //   x + fabs(x) = max(2*x, 0)
    //   (x * 2)^2 / 4 = x^2

    for (index_t d = 0; d < mdim; d++) {
      double v1 = b[d].lo - a[d].hi;
      double v2 = a[d].lo - b[d].hi;

      double v = (v1 + fabs(v1)) + (v2 + fabs(v2));

      sumsq += v * v;
    }

    return sumsq / 4;
  }

  double MinDistanceSqToBoundFarEnd(const StaticHrectBound& other) const {
    double sumsq = 0;
    const DBound *a = this->bounds_;
    const DBound *b = other.bounds_;
    index_t mdim = t_length;

    for (index_t d = 0; d < mdim; d++) {
      double v1 = b[d].hi - a[d].hi;
      double v2 = a[d].lo - b[d].lo;
      
      double v = max(v1, v2);
      v = (v + fabs(v)); /* truncate negative */
      
      sumsq += v * v;
    }

    return sumsq / 4;
  }

  double MaxDistanceSqToBound(const StaticHrectBound& other) const {
    double sumsq = 0;
    const DBound *a = this->bounds_;
    const DBound *b = other.bounds_;

    for (index_t d = 0; d < t_length; d++) {
      double v = max(b[d].hi - a[d].lo, a[d].hi - b[d].lo);
      
      sumsq += v * v;
    }

    return sumsq;
  }

  double MidDistanceSqToBound(const StaticHrectBound& other) const {
    double sumsq = 0;
    const DBound *a = this->bounds_;
    const DBound *b = other.bounds_;
    
    for (index_t d = 0; d < t_length; d++) {
      double v = (a[d].hi + a[d].lo - b[d].hi - b[d].lo) * 0.5;
      
      sumsq += v * v;
    }

    return sumsq;
  }
  
  void Update(const Vector& vector) {
    DEBUG_ASSERT(vector.length() == t_length);
    
    for (index_t i = 0; i < t_length; i++) {
      DBound* bound = &bounds_[i];
      double d = vector[i];
      
      if (unlikely(d > bound->hi)) {
        bound->hi = d;
      }
      if (unlikely(d < bound->lo)) {
        bound->lo = d;
      }
    }
  }
  
  const DBound& get(index_t i) const {
    DEBUG_BOUNDS(i, t_length);
    return bounds_[i];
  }
};

#endif
