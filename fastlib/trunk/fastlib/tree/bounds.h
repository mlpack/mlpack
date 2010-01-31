// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file tree/bounds.h
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * TODO: Come up with a better design so you can do plug-and-play distance
 * metrics.
 *
 * @experimental
 */

#ifndef TREE_BOUNDS_H
#define TREE_BOUNDS_H

#include "fastlib/la/matrix.h"
#include "fastlib/la/la.h"

#include "fastlib/math/math_lib.h"

/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 */
template<int t_pow = 2>
class DHrectBound {
 public:
  static const int PREFERRED_POWER = t_pow;

 private:
  DRange *bounds_;
  index_t dim_;

  OBJECT_TRAVERSAL(DHrectBound) {
    OT_OBJ(dim_);
    OT_ALLOC(bounds_, dim_);
  };

 public:
  /**
   * Initializes to specified dimensionality with each dimension the empty
   * set.
   */
  void Init(index_t dimension) {
    //DEBUG_ASSERT_MSG(dim_ == BIG_BAD_NUMBER, "Already initialized");
    bounds_ = mem::Alloc<DRange>(dimension);

    dim_ = dimension;
    Reset();
  }
  
  /**
   * Makes this (uninitialized) box the average of the two arguments, 
   * i.e. the max and min of each range is the average of the maxes and mins 
   * of the arguments.  
   *
   * Added by: Bill March
   */
  void AverageBoxesInit(const DHrectBound& box1, const DHrectBound& box2) {
  
    index_t dim = box1.dim();
    DEBUG_ASSERT(dim == box2.dim());
    
    Init(dim);
    
    for (index_t i = 0; i < dim; i++) {
    
      DRange range;
      range = box1.get(i) +  box2.get(i);
      range *= 0.5;
      bounds_[i] = range;
    
    } 
    
  } // AverageBoxesInit()

  /** 
   * Computes the weighted average bounding box of two given boxes.
   *
   * If A \in box1 and B \in box2, each point in box1 (box2) is associated 
   * with a weight \alpha (\beta), then this box bounds the vectors 
   * (\alpha A + \beta B)/(\alpha + \beta)
   *
   * alpha1_min and alpha1_max bound the possible (positive) weights of the 
   * vectors in box1
   *
   * added by Bill March 
   */
  void WeightedAverageBoxesInit(double alpha1_min, double alpha1_max, 
                                const DHrectBound& box1, double alpha2_min, 
                                double alpha2_max, const DHrectBound& box2) {
  
    index_t dim = box1.dim();
    DEBUG_ASSERT(dim == box2.dim());
    
    Init(dim);
    
    for (index_t i = 0; i < dim; i++) {
      
      DRange range1 = box1.get(i);
      DRange range2 = box2.get(i);
      
      // account for negative and positive ranges
      double min_x = 0.0;
      if (range1.lo >= 0.0) {
        min_x += alpha1_min * range1.lo; 
      }
      else {
        min_x += alpha1_max * range1.lo;
      }
      
      if (range2.lo >= 0.0) {
        min_x += alpha2_min * range2.lo; 
      }
      else {
        min_x += alpha2_max * range2.lo;
      }

      if (min_x >= 0.0) {
        min_x = min_x/(alpha1_max + alpha2_max);
      }
      else {
        min_x = min_x/(alpha1_min + alpha2_min);
      }
      
      
      double max_x = 0.0;
      if (range1.hi >= 0.0) {
        max_x += alpha1_max * range1.hi;
      }
      else {
        max_x += alpha1_min * range1.hi;
      }
      
      if (range2.hi >= 0.0) {
        max_x += alpha2_max * range2.hi;
      }
      else {
        max_x += alpha2_min * range2.hi;
      }
      
      if (max_x >= 0.0) {
        max_x = max_x/(alpha1_min + alpha2_min);
      }
      else {
        max_x = max_x/(alpha1_max + alpha2_max);
      }
      
      DRange out_range;
      out_range.Init(min_x, max_x);
      
      bounds_[i] = out_range;
      
    } 
  
  } // WeightedAverageBoxesInit()

  /**
   * Resets all dimensions to the empty set.
   */
  void Reset() {
    for (index_t i = 0; i < dim_; i++) {
      bounds_[i].InitEmptySet();
    }
  }

  /**
   * Determines if a point is within this bound.
   */
  bool Contains(const Vector& point) const {
    for (index_t i = 0; i < point.length(); i++) {
      if (!bounds_[i].Contains(point[i])) {
        return false;
      }
    }

    return true;
  }

  bool Contains(const Vector& point, const Vector& box) const {
    const DRange *a = this->bounds_;
    for (index_t i = 0; i < point.length(); i++){
      if (a[i].hi > a[i].lo){
	if (!bounds_[i].Contains(point[i])){
	  return false;
	}
      } else if(point[i] > a[i].hi & point[i] < a[i].lo){
	return false;
      }       
    }
    return true;
  } 


  /** Gets the dimensionality */
  index_t dim() const {
    return dim_;
  }

  /**
   * Gets the range for a particular dimension.
   */
  const DRange& get(index_t i) const {
    DEBUG_BOUNDS(i, dim_);
    return bounds_[i];
  }
  
  /**
   * Calculates the maximum distance within the rectangle
   */
  double CalculateMaxDistanceSq() const {
    double max_distance=0;
    for (index_t i = 0; i < dim_; i++) {
      max_distance+=math::Sqr(bounds_[i].width());
    }
    return max_distance;  
  }
  
  /** Calculates the midpoint of the range */
  void CalculateMidpoint(Vector *centroid) const {
    centroid->Init(dim_);
    for (index_t i = 0; i < dim_; i++) {
      (*centroid)[i] = bounds_[i].mid();
    }
  }
  /** Calculates the midpoint of the range */
  void CalculateMidpointOverwrite(Vector *centroid) const {
    for (index_t i = 0; i < dim_; i++) {
      (*centroid)[i] = bounds_[i].mid();
    }
  }


  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistanceSq(const double *mpoint) const {
    double sum = 0;
    const DRange *mbound = bounds_;

    index_t d = dim_;

    do {
      double v = *mpoint;
      double v1 = mbound->lo - v;
      double v2 = v - mbound->hi;

      v = (v1 + fabs(v1)) + (v2 + fabs(v2));

      mbound++;
      mpoint++;

      sum += math::Pow<t_pow, 1>(v); // v is non-negative
    } while (--d);

    return math::Pow<2, t_pow>(sum) / 4;
  }

 /**
  * Calcualtes minimum bound-to-bound squared distance, with
  * an offset between their respective coordinate systems.
  */
 double MinDistanceSq(const DHrectBound& other, const Vector& offset) const {
   double sum = 0;
   const DRange *a = this->bounds_;
   const DRange *b = other.bounds_;
   index_t mdim = dim_;
   
   DEBUG_SAME_SIZE(dim_, other.dim_);
   //Add Debug for offset vector

   for (index_t d = 0; d < mdim; d++) {
     double v1 = b[d].lo - offset[d] - a[d].hi;
     double v2 = a[d].lo + offset[d] - b[d].lo;

     double v = (v1 + fabs(v1)) + (v2 + fabs(v2));

     sum += math::Pow<t_pow, 1>(v);
   } 

   return math::Pow<2, t_pow>(sum) / 4;
 }

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistanceSq(const Vector& point) const {
    DEBUG_SAME_SIZE(point.length(), dim_);
    return MinDistanceSq(point.ptr());
  }

  /**
   * Calculates minimum bound-to-bound squared distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  double MinDistanceSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;
    index_t mdim = dim_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < mdim; d++) {
      double v1 = b[d].lo - a[d].hi;
      double v2 = a[d].lo - b[d].hi;
      // We invoke the following:
      //   x + fabs(x) = max(x * 2, 0)
      //   (x * 2)^2 / 4 = x^2
      double v = (v1 + fabs(v1)) + (v2 + fabs(v2));

      sum += math::Pow<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum) / 4;
  }


  /**
   * Calculates maximum bound-to-point squared distance.
   */
  double MaxDistanceSq(const Vector& point) const {
    double sum = 0;

    DEBUG_SAME_SIZE(point.length(), dim_);

    for (index_t d = 0; d < dim_; d++) {
      double v = std::max(point[d] - bounds_[d].lo, bounds_[d].hi - point[d]);
      sum += math::Pow<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum);
  }

  /**
   * Calculates maximum bound-to-point squared distance.
   */
  double MaxDistanceSq(const double *point) const {
    double sum = 0;

    for (index_t d = 0; d < dim_; d++) {
      double v = std::max(point[d] - bounds_[d].lo, bounds_[d].hi - point[d]);
      sum += math::Pow<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum);
  }

  /**
   * Computes maximum distance.
   */
  double MaxDistanceSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      double v = std::max(b[d].hi - a[d].lo, a[d].hi - b[d].lo);
      sum += math::PowAbs<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum);
  } 

 /**
  * Computes maximum distance with offset
  */
 double MaxDistanceSq(const DHrectBound& other, const Vector& offset) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      double v = std::max(b[d].hi + offset[d] - a[d].lo, 
			  a[d].hi - offset[d] - b[d].lo);
      sum += math::PowAbs<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum);
  }

 

 // Computes width of box in periodic coordinates
 double width(int dim, double box_width) const {
   double result;
   const DRange *a = this->bounds_;
   result = a[dim].hi-a[dim].lo;
   if (result < 0){
     result = result + box_width;
   }
   return result;
 }


 double MaxDelta(const DHrectBound& other, double box_width, int dim)
 const{
   const DRange *a = this->bounds_; 
   const DRange *b = other.bounds_;
   double result = 0.5*box_width;
   double temp = b[dim].hi - a[dim].lo;
   temp = temp - floor(temp / box_width + 0.5)*box_width;
   if (temp < 0){
     double temp2 = b[dim].lo - a[dim].hi;
     temp2 = temp2 - floor(temp2 / box_width + 0.5)*box_width;
     if (temp2 > 0){
       return result;
     }
   }
   return temp;  
 }

 double MinDelta(const DHrectBound& other, double box_width, int dim)
 const{
   const DRange *a = this->bounds_;   
   const DRange *b  = other.bounds_;
   double result = -0.5*box_width;
   double temp = b[dim].lo - a[dim].hi;
   temp = temp - floor(temp / box_width + 0.5)*box_width;
   if (temp > 0){
     double temp2 = b[dim].hi - a[dim].lo;
     temp2 = temp2 - floor(temp2 / box_width + 0.5)*box_width;
     if (temp2 < 0){
       return result;
     }
   }
   return temp;
 }



  /**
   * Calculates minimum and maximum bound-to-bound squared distance.
   */
  DRange RangeDistanceSq(const DHrectBound& other) const {
    double sum_lo = 0;
    double sum_hi = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;
    index_t mdim = dim_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < mdim; d++) {
      double v1 = b[d].lo - a[d].hi;
      double v2 = a[d].lo - b[d].hi;
      // We invoke the following:
      //   x + fabs(x) = max(x * 2, 0)
      //   (x * 2)^2 / 4 = x^2
      double v_lo = (v1 + fabs(v1)) + (v2 + fabs(v2));
      double v_hi = -std::min(v1, v2);

      sum_lo += math::Pow<t_pow, 1>(v_lo); // v_lo is non-negative
      sum_hi += math::Pow<t_pow, 1>(v_hi); // v_hi is non-negative
    }

    return DRange(math::Pow<2, t_pow>(sum_lo) / 4,
        math::Pow<2, t_pow>(sum_hi));
  }

  /**
   * Calculates minimum and maximum bound-to-point squared distance.
   */
  DRange RangeDistanceSq(const Vector& point) const {
    double sum_lo = 0;
    double sum_hi = 0;
    const double *mpoint = point.ptr();
    const DRange *mbound = bounds_;

    DEBUG_SAME_SIZE(point.length(), dim_);

    index_t d = dim_;
    do {
      double v = *mpoint;
      double v1 = mbound->lo - v;
      double v2 = v - mbound->hi;

      sum_lo += math::Pow<t_pow, 1>((v1 + fabs(v1)) + (v2 + fabs(v2)));
      sum_hi += math::Pow<t_pow, 1>(-std::min(v1, v2));

      mpoint++;
      mbound++;
    } while (--d);

    return DRange(math::Pow<2, t_pow>(sum_lo) / 4,
                  math::Pow<2, t_pow>(sum_hi));
  }

  /**
   * Calculates closest-to-their-midpoint bounding box distance,
   * i.e. calculates their midpoint and finds the minimum box-to-point
   * distance.
   *
   * Equivalent to:
   * <code>
   * other.CalcMidpoint(&other_midpoint)
   * return MinDistanceSqToPoint(other_midpoint)
   * </code>
   */
  double MinToMidSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      double v = b->mid();
      double v1 = a->lo - v;
      double v2 = v - a->hi;

      v = (v1 + fabs(v1)) + (v2 + fabs(v2));

      a++;
      b++;

      sum += math::Pow<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Computes minimax distance, where the other node is trying to avoid me.
   */
  double MinimaxDistanceSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;
    index_t mdim = dim_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < mdim; d++) {
      double v1 = b[d].hi - a[d].hi;
      double v2 = a[d].lo - b[d].lo;
      double v = std::max(v1, v2);
      v = (v + fabs(v)); /* truncate negatives to zero */
      sum += math::Pow<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Calculates midpoint-to-midpoint bounding box distance.
   */
  double MidDistanceSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      sum += math::PowAbs<t_pow, 1>(a[d].hi + a[d].lo - b[d].hi - b[d].lo);
    }

    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Expands this region to include a new point.
   */
  DHrectBound& operator |= (const Vector& vector) {
    DEBUG_SAME_SIZE(vector.length(), dim_);

    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= vector[i];
    }

    return *this;
  }

  /**
   * Expands this region to encompass another bound.
   */
  DHrectBound& operator |= (const DHrectBound& other) {
    DEBUG_SAME_SIZE(other.dim_, dim_);

    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= other.bounds_[i];
    }

    return *this;
  }


 /**
  * Expand this bounding box to encompass another point. Done to 
  * minimize added volume in periodic coordinates.
  */
 DHrectBound& Add(const Vector&  other, const Vector& size){
   DEBUG_SAME_SIZE(other.length(), dim_);
   // Catch case of uninitialized bounds
   if (bounds_[0].hi < -10*size[0]){
     for (index_t i = 0; i < dim_; i++){
       bounds_[i] |= other[i] - floor(other[i] / size[i] + 0.5)*size[i];
     }
   }

   for (index_t i= 0; i < dim_; i++){
     double ah, al;
     ah = bounds_[i].hi - other[i];
     al = bounds_[i].lo - other[i];
     // Project onto range 0 to L
     ah = ah - floor(ah / size[i])*size[i];
     al = al - floor(al / size[i])*size[i];
     if (ah >= al){
       if (size[i] - al < ah){
	 bounds_[i].hi = other[i] - floor(other[i] / size[i] + 0.5)*size[i];
       } else {
	 bounds_[i].lo = other[i] - floor(other[i] / size[i] + 0.5)*size[i];;
       }      
     }    
   }
   return *this;
 }


 /**
  * Expand this bounding box in periodic coordinates, minimizing added volume.
  */
 DHrectBound& Add(const DHrectBound& other, const Vector& size){
   if (bounds_[0].hi < 0){
     for (index_t i = 0; i < dim_; i++){
       bounds_[i] |= other.bounds_[i];
     }
   }

   for (index_t i = 0; i < dim_; i++) {
     double ah, al, bh, bl;
     ah = bounds_[i].hi;
     al = bounds_[i].lo;
     bh = other.bounds_[i].hi;
     bl = other.bounds_[i].lo;
     ah = ah - al;    
     bh = bh - al;
     bl = bl - al;              
     ah = ah - floor(ah / size[i])*size[i];
     bh = bh - floor(bh / size[i])*size[i];
     bl = bl - floor(bl / size[i])*size[i];

     if (((bh > ah) & (bh < bl | ah > bl )) ||
	 (bh >= bl & bl > ah & bh < ah -bl + size[i])){
       bounds_[i].hi = other.bounds_[i].hi -  
	 floor(other.bounds_[i].hi / size[i] + 0.5)*size[i];
     }
    
     if (bl > ah && ((bl > bh) || (bh >= ah -bl + size[i]))){
       bounds_[i].lo = other.bounds_[i].lo -  
	 floor(other.bounds_[i].lo / size[i] + 0.5)*size[i];
     }   

     if (unlikely(ah > bl & bl > bh)){
       bounds_[i].lo = -size[i]/2;
       bounds_[i].hi = size[i]/2;
     }   

   }   
   return *this;
 }



};

/**
 * An L_p metric for vector spaces.
 *
 * A generic Metric class should simply compute the distance between
 * two points.  An LMetric operates for integer powers on Vector spaces.
 */
template<int t_pow>
class LMetric {
 public:
  /**
   * Computes the distance metric between two points.
   */
  static double Distance(const Vector& a, const Vector& b) {
    return math::Pow<1, t_pow>(
        la::RawLMetric<t_pow>(a.length(), a.ptr(), b.ptr()));
  }

  /**
   * Computes the distance metric between two points, raised to a
   * particular power.
   *
   * This might be faster so that you could get, for instance, squared
   * L2 distance.
   */
  template<int t_result_pow>
  static double PowDistance(const Vector& a, const Vector& b) {
    return math::Pow<t_result_pow, t_pow>(
        la::RawLMetric<t_pow>(a.length(), a.ptr(), b.ptr()));
  }
};

/**
 * Ball bound that works in arbitrary metric spaces.
 *
 * See LMetric for an example metric template parameter.
 *
 * To initialize this, set the radius with @c set_radius
 * and set the point by initializing @c point() directly.
 */
template<typename TMetric = LMetric<2>, typename TPoint = Vector>
class DBallBound {
 public:
  typedef TPoint Point;
  typedef TMetric Metric;

 private:
  double radius_;
  TPoint center_;

  OBJECT_TRAVERSAL(DBallBound) {
    OT_OBJ(radius_);
    OT_OBJ(center_);
  }

 public:
  double radius() const {
    return radius_;
  }

  void set_radius(double d) {
    radius_ = d;
  }

  const TPoint& center() const {
    return center_;
  }

  TPoint& center() {
    return center_;
  }

  /**
   * Determines if a point is within this bound.
   */
  bool Contains(const Point& point) const {
    return MidDistance(point) <= radius_;
  }

  /**
   * Gets the center.
   *
   * Don't really use this directly.  This is only here for consistency
   * with DHrectBound, so it can plug in more directly if a "centroid"
   * is needed.
   */
  void CalculateMidpoint(Point *centroid) const {
    ot::InitCopy(centroid, center_);
  }

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistance(const Point& point) const {
    return math::ClampNonNegative(MidDistance(point) - radius_);
  }

  double MinDistanceSq(const Point& point) const {
    return math::Pow<2, 1>(MinDistance(point));
  }

  /**
   * Calculates minimum bound-to-bound squared distance.
   */
  double MinDistance(const DBallBound& other) const {
    double delta = MidDistance(other.center_) - radius_ - other.radius_;
    return math::ClampNonNegative(delta);
  }

  double MinDistanceSq(const DBallBound& other) const {
    return math::Pow<2, 1>(MinDistance(other));
  }

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const Point& point) const {
    return MidDistance(point) + radius_;
  }

  double MaxDistanceSq(const Point& point) const {
    return math::Pow<2, 1>(MaxDistance(point));
  }

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const DBallBound& other) const {
    return MidDistance(other.center_) + radius_ + other.radius_;
  }

  double MaxDistanceSq(const DBallBound& other) const {
    return math::Pow<2, 1>(MaxDistance(other));
  }

  /**
   * Calculates minimum and maximum bound-to-bound squared distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  DRange RangeDistance(const DBallBound& other) const {
    double delta = MidDistance(other.center_);
    double sumradius = radius_ + other.radius_;
    return DRange(
       math::ClampNonNegative(delta - sumradius),
       delta + sumradius);
  }

  DRange RangeDistanceSq(const DBallBound& other) const {
    double delta = MidDistance(other.center_);
    double sumradius = radius_ + other.radius_;
    return DRange(
       math::Pow<2, 1>(math::ClampNonNegative(delta - sumradius)),
       math::Pow<2, 1>(delta + sumradius));
  }

  /**
   * Calculates closest-to-their-midpoint bounding box distance,
   * i.e. calculates their midpoint and finds the minimum box-to-point
   * distance.
   *
   * Equivalent to:
   * <code>
   * other.CalcMidpoint(&other_midpoint)
   * return MinDistanceSqToPoint(other_midpoint)
   * </code>
   */
  double MinToMid(const DBallBound& other) const {
    double delta = MidDistance(other.center_) - radius_;
    return math::ClampNonNegative(delta);
  }

  double MinToMidSq(const DBallBound& other) const {
    return math::Pow<2, 1>(MinToMid(other));
  }

  /**
   * Computes minimax distance, where the other node is trying to avoid me.
   */
  double MinimaxDistance(const DBallBound& other) const {
    double delta = MidDistance(other.center_) + other.radius_ - radius_;
    return math::ClampNonNegative(delta);
  }

  double MinimaxDistanceSq(const DBallBound& other) const {
    return math::Pow<2, 1>(MinimaxDistance(other));
  }

  /**
   * Calculates midpoint-to-midpoint bounding box distance.
   */
  double MidDistance(const DBallBound& other) const {
    return MidDistance(other.center_);
  }

  double MidDistanceSq(const DBallBound& other) const {
    return math::Pow<2, 1>(MidDistance(other));
  }

  double MidDistance(const Point& point) const {
    return Metric::Distance(center_, point);
  }
};

#endif
