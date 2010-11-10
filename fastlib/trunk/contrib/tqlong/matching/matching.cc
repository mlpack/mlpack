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

std::string toString (const Matrix& M)
{
  std::stringstream s;
  for (int i = 0; i < M.n_rows(); i++)
  {
    if (i > 0) s << " ";
    else s << "(";
    for (int j = 0; j < M.n_cols(); j++)
      s << " " << M.get(i, j);
    if (i < M.n_rows()-1) s << "\n";
    else s << ")";
  }
  return s.str();
}

MATCHING_NAMESPACE_END;
