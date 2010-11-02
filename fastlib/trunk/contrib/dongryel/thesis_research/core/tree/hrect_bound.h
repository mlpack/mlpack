/** @file hrect_bound.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_HRECT_BOUND_H
#define CORE_TREE_HRECT_BOUND_H

#include "core/metric_kernels/lmetric.h"
#include "core/math/range.h"

namespace core {
namespace tree {

extern core::table::MemoryMappedFile *global_m_file_;

class HrectBound {
  public:
    static const int t_pow = 2;

  private:
    core::math::Range *bounds_;

    int dim_;

  public:

    HrectBound() {
      dim_ = -1;
      bounds_ = NULL;
    }

    ~HrectBound() {
      if(core::table::global_m_file_) {
        core::table::global_m_file_->Deallocate(bounds_);
      }
      else {
        delete[] bounds_;
      }
    }

    /**
     * Initializes to specified dimensionality with each dimension the empty
     * set.
     */
    void Init(int dimension) {
      bounds_ = (core::table::global_m_file_) ?
                (core::math::Range *) core::table::global_m_file_->Allocate(
                  dimension * sizeof(core::math::Range)) :
                new core::math::Range[dimension];

      dim_ = dimension;
      Reset();
    }

    /**
     * Resets all dimensions to the empty set.
     */
    void Reset() {
      for(int i = 0; i < dim_; i++) {
        bounds_[i].InitEmptySet();
      }
    }

    /**
     * Determines if a point is within this bound.
     */
    bool Contains(const core::table::AbstractPoint &point) const {
      for(int i = 0; i < point.length(); i++) {
        if(!bounds_[i].Contains(point[i])) {
          return false;
        }
      }

      return true;
    }

    /** Gets the dimensionality */
    int dim() const {
      return dim_;
    }

    /**
     * Gets the range for a particular dimension.
     */
    const core::math::Range& get(int i) const {
      return bounds_[i];
    }

    core::math::Range &get(int i) {
      return bounds_[i];
    }

    /**
     * Calculates minimum bound-to-point squared distance.
     */
    double MinDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::AbstractPoint& point) const {

      double sum = 0;
      const core::math::Range *mbound = bounds_;

      for(int i = 0; i < point.length(); i++) {
        double v = point[i];
        double v1 = mbound[i].lo - v;
        double v2 = v - mbound[i].hi;

        v = (v1 + fabs(v1)) + (v2 + fabs(v2));

        sum += core::math::Pow<t_pow, 1>(v); // v is non-negative
      }

      return core::math::Pow<2, t_pow>(sum) / 4;
    }

    /**
     * Calculates minimum bound-to-bound squared distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    double MinDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const HrectBound& other) const {

      double sum = 0;
      const core::math::Range *a = this->bounds_;
      const core::math::Range *b = other.bounds_;

      for(int d = 0; d < dim_; d++) {
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
    double MaxDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::AbstractPoint& point) const {
      double sum = 0;

      for(int d = 0; d < dim_; d++) {
        double v = std::max(point[d] - bounds_[d].lo, bounds_[d].hi - point[d]);
        sum += math::Pow<t_pow, 1>(v); // v is non-negative
      }

      return math::Pow<2, t_pow>(sum);
    }

    /**
     * Computes maximum distance.
     */
    double MaxDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const HrectBound& other) const {
      double sum = 0;
      const core::math::Range *a = this->bounds_;
      const core::math::Range *b = other.bounds_;

      for(int d = 0; d < dim_; d++) {
        double v = std::max(b[d].hi - a[d].lo, a[d].hi - b[d].lo);
        sum += math::PowAbs<t_pow, 1>(v); // v is non-negative
      }

      return math::Pow<2, t_pow>(sum);
    }

    /**
     * Calculates minimum and maximum bound-to-bound squared distance.
     */
    core::math::Range RangeDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const HrectBound &other) const {
      double sum_lo = 0;
      double sum_hi = 0;
      const core::math::Range *a = this->bounds_;
      const core::math::Range *b = other.bounds_;

      for(int d = 0; d < dim_; d++) {
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

      return core::math::Range(math::Pow<2, t_pow>(sum_lo) / 4.0,
                               math::Pow<2, t_pow>(sum_hi));
    }

    /**
     * Calculates minimum and maximum bound-to-point squared distance.
     */
    core::math::Range RangeDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::AbstractPoint& point) const {

      double sum_lo = 0;
      double sum_hi = 0;
      const core::math::Range *mbound = bounds_;

      for(int i = 0; i < dim_; i++) {
        double v = point[i];
        double v1 = mbound[i].lo - v;
        double v2 = v - mbound[i].hi;

        sum_lo += math::Pow<t_pow, 1>((v1 + fabs(v1)) + (v2 + fabs(v2)));
        sum_hi += math::Pow<t_pow, 1>(-std::min(v1, v2));
      }

      return core::math::Range(
               math::Pow<2, t_pow>(sum_lo) / 4.0,
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
    double MinToMidSq(
      const core::metric_kernels::AbstractMetric &metric,
      const HrectBound &other) const {

      double sum = 0;
      const core::math::Range *a = this->bounds_;
      const core::math::Range *b = other.bounds_;

      for(int d = 0; d < dim_; d++) {
        double v = b->mid();
        double v1 = a->lo - v;
        double v2 = v - a->hi;

        v = (v1 + fabs(v1)) + (v2 + fabs(v2));

        a++;
        b++;

        sum += math::Pow<t_pow, 1>(v); // v is non-negative
      }

      return math::Pow<2, t_pow>(sum) / 4.0;
    }

    /**
     * Computes minimax distance, where the other node is trying to avoid me.
     */
    double MinimaxDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const HrectBound &other) const {

      double sum = 0;
      const core::math::Range *a = this->bounds_;
      const core::math::Range *b = other.bounds_;

      for(int d = 0; d < dim_; d++) {
        double v1 = b[d].hi - a[d].hi;
        double v2 = a[d].lo - b[d].lo;
        double v = std::max(v1, v2);
        v = (v + fabs(v)); /* truncate negatives to zero */
        sum += math::Pow<t_pow, 1>(v); // v is non-negative
      }

      return math::Pow<2, t_pow>(sum) / 4.0;
    }

    /**
     * Calculates midpoint-to-midpoint bounding box distance.
     */
    double MidDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const HrectBound &other) const {
      double sum = 0;
      const core::math::Range *a = this->bounds_;
      const core::math::Range *b = other.bounds_;

      for(int d = 0; d < dim_; d++) {
        sum += math::PowAbs<t_pow, 1>(a[d].hi + a[d].lo - b[d].hi - b[d].lo);
      }

      return math::Pow<2, t_pow>(sum) / 4.0;
    }

    /**
     * Expands this region to include a new point.
     */
    HrectBound& operator |= (const core::table::AbstractPoint& vector) {
      for(int i = 0; i < dim_; i++) {
        bounds_[i] |= vector[i];
      }

      return *this;
    }

    /**
     * Expands this region to encompass another bound.
     */
    HrectBound& operator |= (const HrectBound &other) {
      for(int i = 0; i < dim_; i++) {
        bounds_[i] |= other.bounds_[i];
      }

      return *this;
    }
};
};
};

#endif
