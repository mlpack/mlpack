/** @file hrect_bound.h
 *
 *  A declaration of the hyperrectangle bounding primitive.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_HRECT_BOUND_H
#define CORE_TREE_HRECT_BOUND_H

#include <boost/interprocess/offset_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include "core/table/dense_point.h"
#include "core/metric_kernels/lmetric.h"
#include "core/math/range.h"

namespace core {
namespace tree {

extern core::table::MemoryMappedFile *global_m_file_;

class HrectBound {
  public:

    /** @brief Hyperrectangle bound is by default operating under the
     *         L_2 metric.
     */
    static const int t_pow = 2;

  private:

    /** @brief An array of bounds.
     */
    boost::interprocess::offset_ptr<core::math::Range> bounds_;

    /** @brief The dimensionality.
     */
    int dim_;

    // For boost serialization.
    friend class boost::serialization::access;

  public:

    /** @brief Returns whether the bound has been initialized or not.
     */
    bool is_initialized() const {
      return dim_ > 0;
    }

    /** @brief The assignment operator that copies.
     */
    void operator=(const HrectBound &bound_in) {
      bounds_ = new core::math::Range[bound_in.dim()];
      dim_ = bound_in.dim();
      for(int i = 0; i < dim_; i++) {
        bounds_[i].lo = bound_in.get(i).lo;
        bounds_[i].hi = bound_in.get(i).hi;
      }
    }

    /** @brief Serialize the bounding box.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // First the dimensionality.
      ar & dim_;
      for(int i = 0; i < dim_; i++) {
        ar & bounds_[i];
      }
    }

    /** @brief Unserialize the bounding box.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      // Load the dimensionality.
      ar & dim_;

      // Allocate the ranges.
      if(bounds_.get() == NULL) {
        bounds_ = (core::table::global_m_file_) ?
                  core::table::global_m_file_->ConstructArray <
                  core::math::Range > (dim_) :
                  new core::math::Range[dim_];
      }
      for(int i = 0; i < dim_; i++) {
        ar & (bounds_[i]);
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief Prints the hyperrectangle bound.
     */
    void Print() const {
      printf("Hyperrectangle of dimension: %d\n", dim_);
      for(int i = 0; i < dim_; i++) {
        printf("(%g, %g) ", bounds_[i].lo, bounds_[i].hi);
      }
      printf("\n");
    }

    /** @brief The default constructor.
     */
    HrectBound() {
      dim_ = -1;
      bounds_ = NULL;
    }

    /** @brief The destructor.
     */
    ~HrectBound() {
      if(core::table::global_m_file_) {
        core::table::global_m_file_->DestroyPtr(bounds_.get());
      }
      else {
        delete[] bounds_.get();
      }
    }

    /** @brief Generate a random point inside the hyperrectangle with
     *         uniform probability.
     */
    void RandomPointInside(arma::vec *random_point_out) const {
      random_point_out->set_size(dim_);
      for(int i = 0; i < dim_; i++) {
        (*random_point_out)[i] = core::math::Random(
                                   bounds_[i].lo, bounds_[i].hi);
      }
    }

    /** @brief Generate a random point inside the hyperrectangle with
     *         uniform probability.
     */
    void RandomPointInside(core::table::DensePoint *random_point_out) const {
      random_point_out->Init(dim_);
      arma::vec random_point_out_alias(random_point_out->ptr(), dim_, false);
      this->RandomPointInside(&random_point_out_alias);
    }

    /** @brief Initializes to specified dimensionality with each
     *  dimension the empty set.
     */
    void Init(int dimension) {
      if(bounds_.get() == NULL) {
        bounds_ = (core::table::global_m_file_) ?
                  core::table::global_m_file_->ConstructArray<core::math::Range>(
                    dimension) :
                  new core::math::Range[dimension];
      }

      dim_ = dimension;
      Reset();
    }

    /** @brief Copy a given hrect bound.
     */
    void Copy(const HrectBound &other_bound) {
      this->Init(other_bound.dim());
      for(int i = 0; i < dim_; i++) {
        bounds_[i] = other_bound.get(i);
      }
    }

    /** @brief Resets all dimensions to the empty set.
     */
    void Reset() {
      for(int i = 0; i < dim_; i++) {
        bounds_[i].InitEmptySet();
      }
    }

    /** @brief Determines if a point is within this bound.
     */
    template<typename MetricType>
    bool Contains(
      const MetricType &metric_in,
      const core::table::DensePoint &point) const {
      for(int i = 0; i < point.length(); i++) {
        if(! bounds_[i].Contains(point[i])) {
          return false;
        }
      }
      return true;
    }

    /** @brief Determines if another hrect is within this bound.
     */
    template<typename MetricType>
    bool Contains(
      const MetricType &metric_in,
      const core::tree::HrectBound &new_bound) const {
      for(int i = 0; i < dim_; i++) {
        if(! bounds_[i].Contains(new_bound.get(i))) {
          return false;
        }
      }
      return true;
    }

    /** @brief Gets the dimensionality.
     */
    int dim() const {
      return dim_;
    }

    /** @brief Gets the range for a particular dimension.
     */
    const core::math::Range& get(int i) const {
      return bounds_[i];
    }

    /** @brief Gets the range for a particular dimension.
     */
    core::math::Range &get(int i) {
      return bounds_[i];
    }

    /** @brief Calculates minimum bound-to-point squared distance.
     */
    template<typename MetricType>
    double MinDistanceSq(
      const MetricType &metric,
      const core::table::DensePoint& point) const {

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

    /** @brief Calculates minimum bound-to-bound squared distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    template<typename MetricType>
    double MinDistanceSq(
      const MetricType &metric,
      const HrectBound& other) const {

      double sum = 0;
      const core::math::Range *a = this->bounds_.get();
      const core::math::Range *b = other.bounds_.get();

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

    /** @brief Calculates maximum bound-to-point squared distance.
     */
    template<typename MetricType>
    double MaxDistanceSq(
      const MetricType &metric,
      const core::table::DensePoint& point) const {
      double sum = 0;

      for(int d = 0; d < dim_; d++) {
        double v = std::max(point[d] - bounds_[d].lo, bounds_[d].hi - point[d]);
        sum += math::Pow<t_pow, 1>(v); // v is non-negative
      }

      return math::Pow<2, t_pow>(sum);
    }

    /** @brief Computes maximum distance.
     */
    template<typename MetricType>
    double MaxDistanceSq(
      const MetricType &metric,
      const HrectBound& other) const {
      double sum = 0;
      const core::math::Range *a = this->bounds_.get();
      const core::math::Range *b = other.bounds_.get();

      for(int d = 0; d < dim_; d++) {
        double v = std::max(b[d].hi - a[d].lo, a[d].hi - b[d].lo);
        sum += math::PowAbs<t_pow, 1>(v); // v is non-negative
      }

      return math::Pow<2, t_pow>(sum);
    }

    /** @brief Calculates minimum and maximum bound-to-bound squared
     *         distance.
     */
    template<typename MetricType>
    core::math::Range RangeDistanceSq(
      const MetricType &metric,
      const HrectBound &other) const {
      double sum_lo = 0;
      double sum_hi = 0;
      const core::math::Range *a = this->bounds_.get();
      const core::math::Range *b = other.bounds_.get();

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

    /** @brief Calculates minimum and maximum bound-to-point squared
     *         distance.
     */
    template<typename MetricType>
    core::math::Range RangeDistanceSq(
      const MetricType &metric,
      const core::table::DensePoint& point) const {

      double sum_lo = 0;
      double sum_hi = 0;
      const core::math::Range *mbound = bounds_.get();

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

    /** @brief Calculates closest-to-their-midpoint bounding box
     * distance, i.e. calculates their midpoint and finds the minimum
     * box-to-point distance.
     *
     * Equivalent to:
     * <code>
     * other.CalcMidpoint(&other_midpoint)
     * return MinDistanceSqToPoint(other_midpoint)
     * </code>
     */
    template<typename MetricType>
    double MinToMidSq(
      const MetricType &metric,
      const HrectBound &other) const {

      double sum = 0;
      const core::math::Range *a = this->bounds_.get();
      const core::math::Range *b = other.bounds_.get();

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

    /** @brief Computes minimax distance, where the other node is
     *         trying to avoid me.
     */
    template<typename MetricType>
    double MinimaxDistanceSq(
      const MetricType &metric,
      const HrectBound &other) const {

      double sum = 0;
      const core::math::Range *a = this->bounds_.get();
      const core::math::Range *b = other.bounds_.get();

      for(int d = 0; d < dim_; d++) {
        double v1 = b[d].hi - a[d].hi;
        double v2 = a[d].lo - b[d].lo;
        double v = std::max(v1, v2);
        v = (v + fabs(v));
        sum += math::Pow<t_pow, 1>(v); // v is non-negative
      }

      return math::Pow<2, t_pow>(sum) / 4.0;
    }

    /** @brief Calculates midpoint-to-midpoint bounding box distance.
     */
    template<typename MetricType>
    double MidDistanceSq(
      const MetricType &metric,
      const HrectBound &other) const {
      double sum = 0;
      const core::math::Range *a = this->bounds_.get();
      const core::math::Range *b = other.bounds_.get();

      for(int d = 0; d < dim_; d++) {
        sum += math::PowAbs<t_pow, 1>(a[d].hi + a[d].lo - b[d].hi - b[d].lo);
      }

      return math::Pow<2, t_pow>(sum) / 4.0;
    }

    /** @brief Expands this region to include a new point.
     */
    HrectBound& operator |= (const core::table::DensePoint& vector) {
      for(int i = 0; i < dim_; i++) {
        bounds_[i] |= vector[i];
      }
      return *this;
    }

    /** @brief Expands this region to encompass another bound.
     */
    HrectBound& operator |= (const HrectBound &other) {
      for(int i = 0; i < dim_; i++) {
        bounds_[i] |= other.bounds_[i];
      }
      return *this;
    }

    /** @brief Expands this region to include a new point.
     */
    template<typename MetricType>
    HrectBound& Expand(
      const MetricType &metric, const core::table::DensePoint& vector) {
      return this->operator |= (vector);
    }

    /** @brief Expands this region to encompass another bound.
     */
    template<typename MetricType>
    HrectBound& Expand(
      const MetricType &metric, const HrectBound &other) {
      return this->operator |= (other);
    }
};
}
}

#endif
