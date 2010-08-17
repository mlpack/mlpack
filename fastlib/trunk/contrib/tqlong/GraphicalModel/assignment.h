#ifndef ASSIGNMENT_H
#define ASSIGNMENT_H

#include "gm.h"

using namespace std;
BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** An assignment is a map from variables to their values
  */
class Assignment : public Map<const Variable*, Value>
{
public:
  typedef Map<const Variable*, Value> _Base;

  // check if the values are in variables' value set
  bool checkFiniteValueIntegrity() const;

  void print1(const std::string& name = "") const;
  std::string toString() const;

  // check if variable assignments agree with another assignment
  // that is all common variables have same values in (*this) and a
  bool agree(const Assignment& a) const;
};

class AssignmentCompare
{
public:
  bool operator() (const Assignment& lhs, const Assignment& rhs) const;
};

END_GRAPHICAL_MODEL_NAMESPACE;
#endif // ASSIGNMENT_H
