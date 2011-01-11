/** @file range.h
 *
 *  Defines a simple class maintaining an interval.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MATH_RANGE_H
#define CORE_MATH_RANGE_H

#include <boost/serialization/serialization.hpp>
#include <limits>

namespace core {
namespace math {

/** @brief Simple real-valued range.
 */
class Range {
  private:

    // For boost serialization.
    friend class boost::serialization::access;

  public:

    /** @brief The lower bound.
     */
    double lo;

    /** @brief The upper bound.
     */
    double hi;

  public:

    /** @brief Serialize/unserialize the range.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & lo;
      ar & hi;
    }

    Range() {
      InitEmptySet();
    }

    /** Initializes to specified values. */
    Range(double lo_in, double hi_in)
      : lo(lo_in), hi(hi_in) {}

    /** Initialize to an empty set, where lo > hi. */
    void InitEmptySet() {
      lo = std::numeric_limits<double>::max();
      hi = -std::numeric_limits<double>::max();
    }

    /** Initializes to -infinity to infinity. */
    void InitUniversalSet() {
      lo = -std::numeric_limits<double>::max();
      hi = std::numeric_limits<double>::max();
    }

    /** Initializes to a range of values. */
    void Init(double lo_in, double hi_in) {
      lo = lo_in;
      hi = hi_in;
    }

    /**
     * Resets to a range of values.
     *
     * Since there is no dynamic memory this is the same as Init, but calling
     * Reset instead of Init probably looks more similar to surrounding code.
     */
    void Reset(double lo_in, double hi_in) {
      lo = lo_in;
      hi = hi_in;
    }

    /**
     * Gets the span of the range, hi - lo.
     */
    double width() const {
      return hi - lo;
    }

    /**
     * Gets the midpoint of this range.
     */
    double mid() const {
      return (hi + lo) / 2;
    }

    /**
     * Interpolates (factor) * hi + (1 - factor) * lo.
     */
    double interpolate(double factor) const {
      return factor * width() + lo;
    }

    /**
     * Simulate an union by growing the range if necessary.
     */
    const Range& operator |= (double d) {
      if(d < lo) {
        lo = d;
      }
      if(d > hi) {
        hi = d;
      }
      return *this;
    }

    /**
     * Sets this range to include only the specified value, or
     * becomes an empty set if the range does not contain the number.
     */
    const Range& operator &= (double d) {
      if(d > lo) {
        lo = d;
      }
      if(d < hi) {
        hi = d;
      }
      return *this;
    }

    /**
     * Expands range to include the other range.
     */
    const Range& operator |= (const Range& other) {
      if(other.lo < lo) {
        lo = other.lo;
      }
      if(other.hi > hi) {
        hi = other.hi;
      }
      return *this;
    }

    /**
     * Shrinks range to be the overlap with another range, becoming an empty
     * set if there is no overlap.
     */
    const Range& operator &= (const Range& other) {
      if(other.lo > lo) {
        lo = other.lo;
      }
      if(other.hi < hi) {
        hi = other.hi;
      }
      return *this;
    }

    /** Scales upper and lower bounds. */
    friend Range operator - (const Range& r) {
      return Range(-r.hi, -r.lo);
    }

    /** Scales upper and lower bounds. */
    const Range& operator *= (double d) {
      lo *= d;
      hi *= d;
      return *this;
    }

    /** Scales upper and lower bounds. */
    friend Range operator *(const Range& r, double d) {
      return Range(r.lo * d, r.hi * d);
    }

    /** Scales upper and lower bounds. */
    friend Range operator *(double d, const Range& r) {
      return Range(r.lo * d, r.hi * d);
    }

    /** Sums the upper and lower independently. */
    const Range& operator += (const Range& other) {
      lo += other.lo;
      hi += other.hi;
      return *this;
    }

    /** Subtracts from the upper and lower.
     * THIS SWAPS THE ORDER OF HI AND LO, assuming a worst case result.
     * This is NOT an undo of the + operator.
     */
    const Range& operator -= (const Range& other) {
      lo -= other.hi;
      hi -= other.lo;
      return *this;
    }

    /** Adds to the upper and lower independently. */
    const Range& operator += (double d) {
      lo += d;
      hi += d;
      return *this;
    }

    /** Subtracts from the upper and lower independently. */
    const Range& operator -= (double d) {
      lo -= d;
      hi -= d;
      return *this;
    }

    friend Range operator + (const Range& a, const Range& b) {
      Range result;
      result.lo = a.lo + b.lo;
      result.hi = a.hi + b.hi;
      return result;
    }

    friend Range operator - (const Range& a, const Range& b) {
      Range result;
      result.lo = a.lo - b.hi;
      result.hi = a.hi - b.lo;
      return result;
    }

    friend Range operator + (const Range& a, double b) {
      Range result;
      result.lo = a.lo + b;
      result.hi = a.hi + b;
      return result;
    }

    friend Range operator - (const Range& a, double b) {
      Range result;
      result.lo = a.lo - b;
      result.hi = a.hi - b;
      return result;
    }

    /**
     * Takes the maximum of upper and lower bounds independently.
     */
    void MaxWith(const Range& range) {
      if(range.lo > lo) {
        lo = range.lo;
      }
      if(range.hi > hi) {
        hi = range.hi;
      }
    }

    /**
     * Takes the minimum of upper and lower bounds independently.
     */
    void MinWith(const Range& range) {
      if(range.lo < lo) {
        lo = range.lo;
      }
      if(range.hi < hi) {
        hi = range.hi;
      }
    }

    /**
     * Takes the maximum of upper and lower bounds independently.
     */
    void MaxWith(double v) {
      if(v > lo) {
        lo = v;
        if(v > hi) {
          hi = v;
        }
      }
    }

    /**
     * Takes the minimum of upper and lower bounds independently.
     */
    void MinWith(double v) {
      if(v < hi) {
        hi = v;
        if(v < lo) {
          lo = v;
        }
      }
    }

    /**
     * Determines if a point is contained within the range.
     */
    bool Contains(double d) const {
      return d >= lo || d <= hi;
    }
};
};
};

#endif
