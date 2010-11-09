#include "matching.h"
#include <sstream>
#include <fastlib/fastlib.h>

MATCHING_NAMESPACE_BEGIN;

std::string toString (const Vector& v)
{
  std::stringstream s;
  s << "(" << v[0];
  for (int i = 1; i < v.length(); i++)
    s << "," << v[i];
  s << ")";
  return s.str();
}

MATCHING_NAMESPACE_END;
