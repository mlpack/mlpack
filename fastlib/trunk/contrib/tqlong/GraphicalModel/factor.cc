#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

double TableF::trash = 0.0;

void TableF::genAssignments(const Domain& dom, unsigned int index, const Assignment& res, Assignment& temp)
{
  // check if all variables has value
  if (index == dom.size())
  {
    if (index > 0)
      insert(value_type(temp, 1.0));
    return;
  }
  const FiniteVariable* var = (const FiniteVariable*) dom[index];
  Assignment::const_iterator it = res.find(var);
  if (it != res.end())
  {
    int val = (*it).second;
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

/** Generate all possible assignments for a domain that agree with restrict assignment */
TableF::TableF(const Domain& dom, const Assignment& res) : dom_(dom)
{
  Assignment temp;
  for (unsigned int i = 0; i < dom.size(); i++)
    DEBUG_ASSERT(dom[i]->type() == VARIABLE_FINITE);
  genAssignments(dom, 0, res, temp);
}

// remove assignments that do not agree with variable in a assignment
void TableF::restricted(const Assignment& a)
{
  Vector<TableF::iterator> its;
  for (TableF::iterator it = this->begin(); it != this->end(); it++)
  {
    const Assignment& b = it->first;
    if (!b.agree(a)) {
      //        b.print("erase");
      its << it;
    }
  }
  for (Vector<TableF::iterator>::iterator it = its.begin(); it != its.end(); it++)
    this->erase(*it);
}

TableF::factor_value_type& TableF::operator[](const Assignment& a)
{
  trash = 0.0;
  TableF::iterator it = this->find(a);
  // a.print("[]");
  // cout << "not found = " << (it == this->end()) << endl;
  return it == this->end() ? trash : (*it).second;
}

const Domain& TableF::domain() { return dom_; }

void TableF::print(const std::string& name) const
{
  cout << name; if (!name.empty()) cout << " = " << endl;
  for (gm::TableF::const_iterator it = this->begin(); it != this->end(); it++)
  {
    const gm::Assignment& a = it->first;
    double val = it->second;
    cout << val << " <-- "; a.print();
  }
}

TableF::factor_value_type TableF::get(const Assignment& a)
{
  // check if dom is a subset of variables in a
  Assignment temp;
  for (unsigned int i = 0; i < dom_.size(); i++)
  {
    Assignment::const_iterator it = a.find(dom_[i]);
    if (it == a.end()) return 0.0;
    else temp[dom_[i]] = (*it).second;
  }
  return this->operator [](temp);
}

END_GRAPHICAL_MODEL_NAMESPACE;
