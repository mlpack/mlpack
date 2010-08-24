#ifndef FACTOR_TEMPLATE_H
#define FACTOR_TEMPLATE_H

#include <algorithm>
#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** A domain is a list of variables, e.g. x0, x1, ... xn,
  * which could be the arguments of a function or factor
  */
typedef Vector<const Variable*> Domain;

/** A function class which is described by a value table
  * and implemented as a map from variable assignment to double or any number_type
  * that is capable of =, +, +=, -, -=, *, *=, /, /=. Example:
  for (TableF<double>::iterator it = f.begin(); it != f.end(); it++)
  {
    const Assignment& assignment = it->first;
    TableF::factor_value_type value = it->second; // double
  }
  */
template <typename _V>
class TableF : public Map<Assignment, _V, AssignmentCompare>
{
  TableF() {}  // hide the default constructor
public:
  typedef Map<Assignment, _V, AssignmentCompare>  _Base;
  typedef _V                                      factor_value_type;

  /** Generate all possible assignments for a domain that agree with restrict assignment */
  TableF(const std::string& name, const Domain& dom, const Assignment& res = Assignment());
  TableF(const TableF& f, const Assignment &res = Assignment());

  /** remove assignments that do not agree with variable in a assignment */
  void restricted(const Assignment& a);

  /** For operator[](a), the assignment must be in the current set of assignments
    * otherwise, the result is a double reference to 0.0
    */
  factor_value_type& operator[](const Assignment& a);

  /** For get(a), the assignment could be a superset of the domain */
  factor_value_type get(const Assignment& a) const;

  /** Return the domain of this factor */
  const Domain& domain() const;

  /** Returnn all assignments of this factor */
  const List<Assignment>& assignments() const { return assignments_; }

  void print1(bool detail = false) const;
  std::string toString(bool detail = false) const;
protected:
  static factor_value_type trash;
  std::string name_;
  Domain dom_;
  List<Assignment> assignments_;

  /** Helper function to generate all assignments agree with res within the domain */
  void genAssignments(const Domain& dom, unsigned int index, const Assignment& res, Assignment& temp);
};

template <typename _V> typename TableF<_V>::factor_value_type TableF<_V>::trash = 0.0;

template <typename _V>
void TableF<_V>::genAssignments(const Domain& dom, unsigned int index, const Assignment& res, Assignment& temp)
{
  // check if all variables has value
  if (index == dom.size())
  {
    if (index > 0)
    {
      insert(typename TableF::value_type(temp, typename TableF::factor_value_type(1.0)));
      assignments_ << temp;
    }
    return;
  }
  // populate all values of variable dom[index] that agree with res
  const Variable* var = (const Variable*) dom[index];
  Assignment::const_iterator it = res.find(var);
  if (it != res.end())
  {
    Value val = (*it).second;
    DEBUG_ASSERT(val >= 0 && val < var->cardinality());
    temp[var] = val;
    genAssignments(dom, index+1, res, temp);
  }
  else
  {
    for (int val = 0; val < var->cardinality(); val++)
    {
      temp[var] = val;
      genAssignments(dom, index+1, res, temp);
    }
  }
}

// Generate all possible assignments for a domain that agree with restrict assignment
template <typename _V>
TableF<_V>::TableF(const std::string& name, const Domain& dom, const Assignment& res) : name_(name), dom_(dom)
{
  Assignment temp;
  BOOST_FOREACH(const Variable* v, dom)
    DEBUG_ASSERT(v->type() == VARIABLE_FINITE);
  genAssignments(dom, 0, res, temp);
}

template <typename _V>
TableF<_V>::TableF(const TableF &f, const Assignment &res)
  : Map<Assignment, _V, AssignmentCompare>(f), name_(f.name_), dom_(f.dom_), assignments_(f.assignments_)
{
  restricted(res);
}

struct RemoveCondition
{
  const Assignment& a_;
  RemoveCondition(const Assignment& a) : a_(a) {}
  bool operator() (const Assignment& b) const { return !b.agree(a_); }
};

template <typename _V>
void TableF<_V>::restricted(const Assignment& a)
{
  for (typename TableF::iterator it = this->begin(); it != this->end();)
  {
    const Assignment& b = it->first;
    if (!b.agree(a))
      this->erase(it++);
    else
      it++;
  }
  assignments_.remove_if(RemoveCondition(a));
  // for (typename List<Assignment>::iterator it = assignments_.begin(); it != assignments_.end();)
  // {
  //   const Assignment& b = (*it);
  //   if (!b.agree(a))
  //     assignments_.erase(it++);
  //   else
  //     it++;
  // }
}

template <typename _V>
typename TableF<_V>::factor_value_type& TableF<_V>::operator[](const Assignment& a)
{
  trash = typename TableF::factor_value_type(0.0);
  typename TableF::iterator it = this->find(a);
  return it == this->end() ? trash : (*it).second;
}

template <typename _V> const Domain& TableF<_V>::domain() const { return dom_; }

template <typename _V> std::string TableF<_V>::toString(bool detail) const
{
  std::ostringstream cout;
  cout << "Factor " << name_;
  if (!detail) return cout.str();
  if (!name_.empty()) cout << " = " << endl;
  unsigned int i = 0;
  BOOST_FOREACH(const typename TableF::value_type& p, *this)
  {
    const gm::Assignment& a = p.first;
    const factor_value_type&  val = p.second;
    cout << val << " <-- " << a.toString();
    if (++i < this->size()) cout << endl;
  }
  return cout.str();
}

template <typename _V>
typename TableF<_V>::factor_value_type TableF<_V>::get(const Assignment& a) const
{
  // check if dom is a subset of variables in a
  Assignment temp;
  BOOST_FOREACH(const Variable* v, dom_)
  {
    Assignment::const_iterator it = a.find(v);
    if (it == a.end()) return 0.0;
    else temp[v] = (*it).second;
  }
  return this->Map<Assignment, _V, AssignmentCompare>::get(temp);
}

END_GRAPHICAL_MODEL_NAMESPACE;
#endif // FACTOR_TEMPLATE_H
