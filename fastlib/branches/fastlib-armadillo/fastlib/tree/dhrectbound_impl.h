/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file tree/dhrectbound_impl.h
 *
 * Implementation of hyper-rectangle bound policy class.
 * Template parameter t_pow is the metric to use; use 2 for Euclidian (L2).
 *
 * @experimental
 */

#ifndef TREE_DHRECTBOUND_IMPL_H
#define TREE_DHRECTBOUND_IMPL_H

/**
 * Initializes to specified dimensionality with each dimension the empty
 * set.
 */
template<int t_pow>
void DHrectBound<t_pow>::Init(index_t dimension) {
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
 * Added by: Bill March, 5/7
 */
template<int t_pow>
void DHrectBound<t_pow>::AverageBoxesInit(const DHrectBound& box1, const DHrectBound& box2) {

  index_t dim = box1.dim();
  DEBUG_ASSERT(dim == box2.dim());

  Init(dim);

  for (index_t i = 0; i < dim; i++) {

    DRange range;
    range = box1.get(i) +  box2.get(i);
    range *= 0.5;
    bounds_[i] = range;

  } 

} // AverageBoxes()

/**
 * Resets all dimensions to the empty set.
 */
template<int t_pow>
void DHrectBound<t_pow>::Reset() {
  for (index_t i = 0; i < dim_; i++) {
    bounds_[i].InitEmptySet();
  }
}

/**
 * Determines if a point is within this bound.
 */
template<int t_pow>
bool DHrectBound<t_pow>::Contains(const Vector& point) const {
  for (index_t i = 0; i < point.length(); i++) {
    if (!bounds_[i].Contains(point[i])) {
      return false;
    }
  }

  return true;
}

template<int t_pow>
bool DHrectBound<t_pow>::Contains(const Vector& point, const Vector& box) const {
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


/**
 * Gets the range for a particular dimension.
 */
template<int t_pow>
const DRange& DHrectBound<t_pow>::get(index_t i) const {
  DEBUG_BOUNDS(i, dim_);
  return bounds_[i];
}

/**
 * Calculates the maximum distance within the rectangle
 */
template<int t_pow>
double DHrectBound<t_pow>::CalculateMaxDistanceSq() const {
  double max_distance = 0;
  for (index_t i = 0; i < dim_; i++) {
    max_distance+=math::Sqr(bounds_[i].width());
  }
  return max_distance;  
}

/** Calculates the midpoint of the range */
template<int t_pow>
void DHrectBound<t_pow>::CalculateMidpoint(Vector *centroid) const {
  centroid->Init(dim_);
  for (index_t i = 0; i < dim_; i++) {
    (*centroid)[i] = bounds_[i].mid();
  }
}
/** Calculates the midpoint of the range */
template<int t_pow>
void DHrectBound<t_pow>::CalculateMidpointOverwrite(Vector *centroid) const {
  for (index_t i = 0; i < dim_; i++) {
    (*centroid)[i] = bounds_[i].mid();
  }
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<int t_pow>
double DHrectBound<t_pow>::MinDistanceSq(const double *mpoint) const {
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
template<int t_pow>
double DHrectBound<t_pow>::MinDistanceSq(const DHrectBound& other, const Vector& offset) const {
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
template<int t_pow>
double DHrectBound<t_pow>::MinDistanceSq(const Vector& point) const {
  DEBUG_SAME_SIZE(point.length(), dim_);
  return MinDistanceSq(point.ptr());
}

/**
 * Calculates minimum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistanceSq(other) for minimum squared distance.
 */
template<int t_pow>
double DHrectBound<t_pow>::MinDistanceSq(const DHrectBound& other) const {
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
template<int t_pow>
double DHrectBound<t_pow>::MaxDistanceSq(const Vector& point) const {
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
template<int t_pow>
double DHrectBound<t_pow>::MaxDistanceSq(const double *point) const {
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
template<int t_pow>
double DHrectBound<t_pow>::MaxDistanceSq(const DHrectBound& other) const {
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
template<int t_pow>
double DHrectBound<t_pow>::MaxDistanceSq(const DHrectBound& other, const Vector& offset) const {
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

/**
 * Computes minimum distance between boxes in periodic coordinate system
 */
template<int t_pow>
double DHrectBound<t_pow>::PeriodicMinDistanceSq(const DHrectBound& other, const Vector& box_size) const {
  double sum = 0;
  const DRange *a = this->bounds_;
  const DRange *b = other.bounds_;

  DEBUG_SAME_SIZE(dim_, other.dim_);

  for (index_t d = 0; d < dim_; d++){      
    double v = 0, d1, d2, d3;
    d1 = (a[d].hi > a[d].lo | b[d].hi > b[d].lo)*
      min(b[d].lo - a[d].hi, a[d].lo-b[d].hi);
    d2 = (a[d].hi > a[d].lo & b[d].hi > b[d].lo)*
      min(b[d].lo - a[d].hi, a[d].lo-b[d].hi + box_size[d]);
    d3 = (a[d].hi > a[d].lo & b[d].hi > b[d].lo)*
      min(b[d].lo - a[d].hi + box_size[d], a[d].lo-b[d].hi);
    v = (d1 + fabs(d1)) + (d2+  fabs(d2)) + (d3 + fabs(d3));
    sum += math::Pow<t_pow, 1>(v);
  }   
  return math::Pow<2, t_pow>(sum) / 4;
}

template<int t_pow>
double DHrectBound<t_pow>::PeriodicMinDistanceSq(const Vector& point, const Vector& box_size) const {
  double sum = 0;  
  const DRange *b = this->bounds_;

  for (index_t d = 0; d < dim_; d++){
    double a = point[d];
    double v = 0, bh;
    bh = b[d].hi-b[d].lo;
    bh = bh - floor(bh / box_size[d])*box_size[d];
    a = a - b[d].lo;
    a = a - floor(a / box_size[d])*box_size[d];
    if (bh > a){
      v = min(a - bh, box_size[d]-a);
    }
    sum += math::Pow<t_pow, 1>(v);
  }

  return math::Pow<2, t_pow>(sum);
}

/**
 * Computes maximum distance between boxes in periodic coordinate system
 */
template<int t_pow>
double DHrectBound<t_pow>::PeriodicMaxDistanceSq(const DHrectBound& other, const Vector& box_size) const {
  double sum = 0;
  const DRange *a = this->bounds_;
  const DRange *b = other.bounds_;

  DEBUG_SAME_SIZE(dim_, other.dim_);

  for (index_t d = 0; d < dim_; d++){
    double v = box_size[d] / 2.0;
    double dh, dl;
    dh = a[d].hi - b[d].lo;
    dh = dh - floor(dh / box_size[d])*box_size[d];
    dl = b[d].hi - a[d].lo;
    dl = dl - floor(dl / box_size[d])*box_size[d];
    v = max(min(dh,v), min(dl, v));

    sum += math::PowAbs<t_pow, 1>(v);
  }   
  return math::Pow<2, t_pow>(sum);
}

template<int t_pow>
double DHrectBound<t_pow>::PeriodicMaxDistanceSq(const Vector& point, const Vector& box_size) const {
  double sum = 0;
  const DRange *a = this->bounds_;  
  for (index_t d = 0; d < dim_; d++){
    double b = point[d];
    double v = box_size[d] / 2.0;
    double ah, al;
    ah = a[d].hi - b;
    ah = ah - floor(ah / box_size[d])*box_size[d];
    if (ah < v){
      v = ah;
    } else {
      al = a[d].lo - b;
      al = al - floor(al / box_size[d])*box_size[d];
      if (al > v){
        v = 2*v-al;
      }
    }
    sum += math::PowAbs<t_pow, 1>(v);
  }   
  return math::Pow<2, t_pow>(sum);
}


template<int t_pow>
double DHrectBound<t_pow>::MaxDelta(const DHrectBound& other, double box_width, int dim) const {
  const DRange *a = this->bounds_; 
  const DRange *b = other.bounds_;
  double result = 0.5*box_width;
  double temp = b[dim].hi - a[dim].lo;
  temp = temp - floor(temp / box_width)*box_width;
  if (temp > box_width / 2){
    temp = b[dim].lo - a[dim].hi;
    temp = temp - floor(temp / box_width)*box_width;
    if (temp > box_width / 2){
      result = b[dim].hi - a[dim].lo;
      result = result - floor(temp / box_width + 1)*box_width;
    } 
  } else {
    result = temp;
  }
  return result;
}

template<int t_pow>
double DHrectBound<t_pow>::MinDelta(const DHrectBound& other, double box_width, int dim) const {
  const DRange *a = this->bounds_;   
  const DRange *b  = other.bounds_;
  double result = -0.5 * box_width;
  double temp = b[dim].hi - a[dim].lo;
  temp = temp - floor(temp/  box_width)*box_width;
  if (temp > box_width / 2){
    temp = b[dim].hi - a[dim].hi;
    temp = temp - floor(temp / box_width)*box_width;
    if (temp > box_width / 2){
      result = temp - box_width;
    } 
  } else {
    temp = b[dim].hi - a[dim].hi;
    result = temp - floor(temp / box_width)*box_width;
  }
  return result;
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 */
template<int t_pow>
DRange DHrectBound<t_pow>::RangeDistanceSq(const DHrectBound& other) const {
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
template<int t_pow>
DRange DHrectBound<t_pow>::RangeDistanceSq(const Vector& point) const {
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
template<int t_pow>
double DHrectBound<t_pow>::MinToMidSq(const DHrectBound& other) const {
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
template<int t_pow>
double DHrectBound<t_pow>::MinimaxDistanceSq(const DHrectBound& other) const {
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
template<int t_pow>
double DHrectBound<t_pow>::MidDistanceSq(const DHrectBound& other) const {
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
template<int t_pow>
DHrectBound<t_pow>& DHrectBound<t_pow>::operator|=(const Vector& vector) {
  DEBUG_SAME_SIZE(vector.length(), dim_);

  for (index_t i = 0; i < dim_; i++) {
    bounds_[i] |= vector[i];
  }

  return *this;
}

/**
 * Expands this region to encompass another bound.
 */
template<int t_pow>
DHrectBound<t_pow>& DHrectBound<t_pow>::operator|=(const DHrectBound& other) {
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
template<int t_pow>
DHrectBound<t_pow>& DHrectBound<t_pow>::Add(const Vector& other, const Vector& size) {
  DEBUG_SAME_SIZE(other.length(), dim_);
  // Catch case of uninitialized bounds
  if (bounds_[0].hi < 0){
    for (index_t i = 0; i < dim_; i++){
      bounds_[i] |= other[i];
    }
  }

  for (index_t i= 0; i < dim_; i++){
    double ah, al;
    ah = bounds_[i].hi - other[i];
    al = bounds_[i].lo - other[i];
    ah = ah - floor(ah / size[i])*size[i];
    al = al - floor(al / size[i])*size[i];
    if (ah < al){
      if (size[i] - ah < al){
        bounds_[i].hi = other[i];
      } else {
        bounds_[i].lo = other[i];
      }
    }     
  }
  return *this;
}

/**
 * Expand this bounding box in periodic coordinates, minimizing added volume.
 */
template<int t_pow>
DHrectBound<t_pow>& DHrectBound<t_pow>::Add(const DHrectBound& other, const Vector& size){
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
      bounds_[i].hi = other.bounds_[i].hi;
    }

    if (bl > ah && ((bl > bh) || (bh >= ah -bl + size[i]))){
      bounds_[i].lo = other.bounds_[i].lo;
    }   

    if (unlikely(ah > bl & bl > bh)){
      bounds_[i].lo = 0;
      bounds_[i].hi = size[i];
    }
  }

  return *this;
}

#endif
