#ifndef FACTOR_H
#define FACTOR_H

#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/**
  * A function class which is described by a value table
  * and implemented as a map from variable assignment to double. Example
  for (TableF::iterator it = f.begin(); it != f.end(); it++)
  {
    const Assignment& assignment = it->first;
    double value = it->second;
  }
  */
class TableF : public Map<Assignment, double, AssignmentCompare>
{
  TableF() {}  // hide the default constructor
public:
  typedef double factor_value_type;

  /** Generate all possible assignments for a domain that agree with restrict assignment */
  TableF(const Domain& dom, const Assignment& res = Assignment());

  // remove assignments that do not agree with variable in a assignment
  void restricted(const Assignment& a);

  // For operator[](a), the assignment must be in the current set of assignments
  // otherwise, the result is a double reference to 0.0
  factor_value_type& operator[](const Assignment& a);

  // For get(a), the assignment could be a superset of the domain
  factor_value_type get(const Assignment& a);

  const Domain& domain();

  void print(const std::string& name = "") const;
protected:
  static double trash;
  Domain dom_;

  void genAssignments(const Domain& dom, unsigned int index, const Assignment& res, Assignment& temp);
};

END_GRAPHICAL_MODEL_NAMESPACE;
#endif // FACTOR_H
