#ifndef VALUE_H
#define VALUE_H

#include <sstream>
#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/**
  * Value class, could be int or double
  */

typedef double Value;

//class Value
//{
//protected:
//  double dblVal;
//public:
//  Value() { *this = 0; }
//  Value(int val) { *this = val; }
//  Value(double val) { *this = val; }
//
//  operator int() const { return (int) dblVal; }
//  operator double() const { return dblVal; }
//
//  int operator = (int val) { dblVal = val; return val; }
//  double operator = (double val) { dblVal = val; return val; }
//
//  friend class ValueCompare;
//};
#define FINITE_VALUE(x) ((int) (x))
#define CONTINUOUS_VALUE(x) ((double)(x))
class ValueCompare
{
public:
  bool operator() (const Value& lhs, const Value& rhs) const
  {
//    return lhs.dblVal < rhs.dblVal;
    return lhs < rhs;
  }
};

END_GRAPHICAL_MODEL_NAMESPACE;

#endif // VALUE_H
