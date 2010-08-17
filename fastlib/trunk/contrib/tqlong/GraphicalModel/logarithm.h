#ifndef LOGARITHM_GRAPH_H
#define LOGARITHM_GRAPH_H

#include "gm.h"
#include <cmath>
#include <iostream>
#include <limits>

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** Implement a nonnegative number by its logarithm (natural base)
  */
class Logarithm
{
public:
  typedef double                        value_type;
protected:
  value_type logValue_;
public:
  Logarithm(const value_type& originalValue = value_type(0))
  {
    DEBUG_ASSERT_MSG(originalValue >= value_type(0), "Must be a nonnegative number");
    logValue_ = log(originalValue);
  }

  Logarithm(const value_type& logValue, int) : logValue_(logValue) {}
  Logarithm(const Logarithm& x) : logValue_(x.logValue_) {}

  // we may not allow automatic type conversion, use toDouble() instead
  // as this may be dangerously implicit, see toDouble() below
  // operator value_type() const { return originalValue(); }

  value_type logValue() const { return logValue_; }
  value_type originalValue() const { return exp(logValue_); }

  Logarithm operator= (Logarithm x) { logValue_ = x.logValue_; return *this; }

  Logarithm operator+ (Logarithm x) const;
  Logarithm operator+= (Logarithm x);

  Logarithm operator- (Logarithm x) const;
  Logarithm operator-= (Logarithm x);

  Logarithm operator* (Logarithm x) const;
  Logarithm operator*= (Logarithm x);

  Logarithm operator/ (Logarithm x) const;
  Logarithm operator/= (Logarithm x);

  bool operator== (Logarithm x) const { return fabs(logValue_ - x.logValue_) < 1e-12; }
  bool operator!= (Logarithm x) const { return !(*this == x); }
  bool operator> (Logarithm x) const { return *this != x && logValue_ > x.logValue_; }
  bool operator>= (Logarithm x) const { return logValue_ >= x.logValue_; }
  bool operator< (Logarithm x) const { return *this != x && logValue_ < x.logValue_; }
  bool operator<= (Logarithm x) const { return logValue_ <= x.logValue_; }
};

END_GRAPHICAL_MODEL_NAMESPACE;

ostream& operator << (ostream& os, const GM_NAMESPACE::Logarithm& x);
GM_NAMESPACE::Logarithm log(const GM_NAMESPACE::Logarithm& x);
double toDouble(double x);
double toDouble(GM_NAMESPACE::Logarithm x);

#endif
