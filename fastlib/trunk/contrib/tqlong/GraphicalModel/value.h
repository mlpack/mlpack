#ifndef VALUE_H
#define VALUE_H

#include <sstream>
#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** Value class, could be int or double
  */
typedef double Value;

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
