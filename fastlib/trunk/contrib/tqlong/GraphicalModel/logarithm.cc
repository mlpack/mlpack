#include "gm.h"

ostream& operator << (ostream& os, const GM_NAMESPACE::Logarithm& x)
{
  return os << "exp(" << x.logValue() << ") = " << x.originalValue();
}

GM_NAMESPACE::Logarithm log(const GM_NAMESPACE::Logarithm& x)
{
  return GM_NAMESPACE::Logarithm(x.logValue());
}
