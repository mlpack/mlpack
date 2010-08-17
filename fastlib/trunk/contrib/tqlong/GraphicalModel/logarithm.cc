#include "gm.h"
#include <boost/math/special_functions/log1p.hpp>

ostream& operator << (ostream& os, const GM_NAMESPACE::Logarithm& x)
{
  return os << "[exp(" << x.logValue() << ") = " << x.originalValue() << "]";
}

double toDouble(double x)
{
  return x;
}

double toDouble(GM_NAMESPACE::Logarithm x)
{
  return x.originalValue();
}

GM_NAMESPACE::Logarithm log(const GM_NAMESPACE::Logarithm& x)
{
  return GM_NAMESPACE::Logarithm(x.logValue());
}

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

Logarithm Logarithm::operator+ (Logarithm x) const
{
  Logarithm result;
  if (logValue_ == -numeric_limits<double>::infinity()) result = x;
  else
    if (x.logValue_ == -numeric_limits<double>::infinity()) result = *this;
  else
    result = (logValue_ > x.logValue_) ?
      Logarithm(logValue_ + log1p( exp(x.logValue_-logValue_) ), 1) :
      Logarithm(x.logValue_ + log1p( exp(logValue_-x.logValue_) ), 1);
//  cout << "operator+ " << *this << " + " << x << " = " << result << endl;
  return result;
}

Logarithm Logarithm::operator+= (Logarithm x)
{
  Logarithm result;
  if (logValue_ == -numeric_limits<double>::infinity()) result = x;
  else
    if (x.logValue_ == -numeric_limits<double>::infinity()) result = *this;
  else
    result = (logValue_ > x.logValue_) ?
      Logarithm(logValue_ + log1p( exp(x.logValue_-logValue_) ), 1) :
      Logarithm(x.logValue_ + log1p( exp(logValue_-x.logValue_) ), 1);
//  cout << "operator+= " << *this << " + " << x << " = " << result << endl;
  *this = result;
  return result;
}

Logarithm Logarithm::operator- (Logarithm x) const
{
  DEBUG_ASSERT_MSG(logValue_ >= x.logValue_, "(*this) must be at least x");
  if (logValue_ == -numeric_limits<double>::infinity()) return *this;
  return Logarithm(logValue_ + log1p( -exp(x.logValue_-logValue_) ), 1);
}

Logarithm Logarithm::operator-= (Logarithm x)
{
  DEBUG_ASSERT_MSG(logValue_ >= x.logValue_, "(*this) must be at least x");
  if (logValue_ != -numeric_limits<double>::infinity());
    logValue_  = logValue_ + log1p( -exp(x.logValue_-logValue_) );
  return *this;
}

Logarithm Logarithm::operator* (Logarithm x) const
{
  Logarithm result(logValue_+x.logValue_, 1);
//  cout << "operator*= " << *this << " + " << x << " = " << result << endl;
  return result;
}

Logarithm Logarithm::operator*= (Logarithm x)
{
  Logarithm result(logValue_+x.logValue_, 1);
//  cout << "operator*= " << *this << " + " << x << " = " << result << endl;
  *this = result;
  return result;
}

Logarithm Logarithm::operator/= (Logarithm x)
{
  DEBUG_ASSERT_MSG(x.logValue_ != -numeric_limits<double>::infinity(), "Divide by zero");
  Logarithm result(logValue_-x.logValue_, 1);
//  cout << "operator/= " << *this << " + " << x << " = " << result << endl;
  *this = result;
  return result;
}

Logarithm Logarithm::operator/ (Logarithm x) const
{
  DEBUG_ASSERT_MSG(x.logValue_ != -numeric_limits<double>::infinity(), "Divide by zero");
  Logarithm result(logValue_-x.logValue_, 1);
//  cout << "operator/= " << *this << " + " << x << " = " << result << endl;
  return result;
}

END_GRAPHICAL_MODEL_NAMESPACE;
