#ifndef UNIVERSE_H
#define UNIVERSE_H

#include <iostream>
#include "gm.h"

using namespace std;
BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/**
  * The universe is a collection of random variables
  */
class Universe : public Set<Variable*>
{
  std::multimap<std::string, Variable*> nameMap;
public:
  typedef std::multimap<std::string, Variable*>::iterator name_iterator;
  typedef std::pair<name_iterator, name_iterator> find_name_result_type;

  Variable* addVariable(Variable* v)
  {
    DEBUG_ASSERT(v);
    (*this) << v;
    nameMap.insert(std::pair<std::string, Variable*>(v->name(), v));
  }

  /** add a new finite variable to the universe */
  Variable* newFiniteVariable(const std::string& name, int cardinality)
  {
    Variable* var = new FiniteVar<int>(name, cardinality);
    (*this) << var;
    nameMap.insert(std::pair<std::string, Variable*>(name, var));
    return var;
  }

  Variable* newVariable(const std::string& name, const FiniteVar<int>& temp)
  {
    Variable* var = new FiniteVar<int>(name, temp);
    (*this) << var;
    nameMap.insert(std::pair<std::string, Variable*>(name, var));
    return var;
  }

  Variable* newVariable(const std::string& name, const FiniteVar<double>& temp)
  {
    Variable* var = new FiniteVar<double>(name, temp);
    (*this) << var;
    nameMap.insert(std::pair<std::string, Variable*>(name, var));
    return var;
  }

  Variable* newVariable(const std::string& name, const FiniteVar<std::string>& temp)
  {
    Variable* var = new FiniteVar<std::string>(name, temp);
    (*this) << var;
    nameMap.insert(std::pair<std::string, Variable*>(name, var));
    return var;
  }

  /** find variable by name, example:
    *   find_result_type result = u.findByName(name);
    *   for (Universe::iterator i = result.first; i != result.second; i++)
    *   {
    *      std::string name = i->first;
    *      Variable* var = i->second;
    *   }
    */
  find_name_result_type findByName(const std::string& name)
  {
    return nameMap.equal_range(name);
  }

  void print(const std::string& name = "") const
  {
    cout << name; if (!name.empty()) cout << " = " << endl;
    for (gm::Universe::const_iterator i = this->begin(); i != this->end(); i++)
    {
      const gm::Variable* var = (*i);
      var->print();
      cout << endl;
//      std::cout << "name = " << var->name();
//      if (var->type() == 0) cout << " cardinality = " << ((gm::FiniteVariable*) var)->cardinality() << endl;
//      else cout << endl;
    }
  }
};

END_GRAPHICAL_MODEL_NAMESPACE;
#endif // UNIVERSE_H
