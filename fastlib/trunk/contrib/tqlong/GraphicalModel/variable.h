#ifndef VARIABLE_H
#define VARIABLE_H

#include "gm.h"
#include <iostream>
using namespace std;
BEGIN_GRAPHICAL_MODEL_NAMESPACE;
/** A basic class represents a random variable
  */
#define VARIABLE_FINITE 0
#define VARIABLE_CONTINUOUS 1
class Variable
{
protected:
  std::string name_;
  /** Random variable type:  0 (discrete/finite), 1 (continuous) */
  int type_;
public:
  Variable(const std::string& name = "", int type = 0) : name_(name), type_(0) {}
  virtual ~Variable() {}
  int type() const { return type_; }
  const std::string& name() const { return name_; }
  virtual int cardinality() const = 0;
  virtual void print1(bool detail = false) const
  {
    cout << "Variable " << name_ << " type = " << type_;
  }
  virtual std::string toString(bool detail = false) const
  {
    std::ostringstream cout;
    cout << "Variable " << name_ << " type = " << type_;
    return cout.str();
  }
};

/** A discrete random variable implementation
  * a variable can only be assigned value from a finite set of values. Usage:

  //First define variable templates
  FiniteVar<int> tmp1("tmp1", 2);
  FiniteVar<string> tmp2("tmp2", valueMap); // DualMap<int, string> valueMap;
  //then add new variables to the universe using templates
  FiniteVariable *x = u.newFiniteVariable("x", tmp1),
                 *y = u.newFiniteVariable("y", tmp1),
                 *z = u.newFiniteVariable("z", tmp2);

  x, y, z will have same cardinality_, and valueMap_ as tmp1, tmp2
  while having their own names.
  */
template <typename _V>  class FiniteVar : public Variable
{
public:
  typedef Variable                   _Base;
  typedef _V                         variable_value_type;
  typedef DualMap<int, _V>           int_value_map_type;
public:
  /** Create a finite variable with its cardinality */
  FiniteVar(const std::string& name, int cardinality)
    : Variable(name, 0), cardinality_(cardinality), valueMap_(NULL)
  {
    DEBUG_ASSERT(cardinality > 0);
  }
  /** Copy another variable, only change the name */
  FiniteVar(const std::string &name, const FiniteVar& var)
    : Variable(name, 0), cardinality_(var.cardinality_), valueMap_(var.valueMap_) { }

  /** Constructor using name & valueMap */
  FiniteVar(const std::string &name, int_value_map_type& valueMap)
    : Variable(name, 0), cardinality_(valueMap.size()), valueMap_(&valueMap)
  {
    DEBUG_ASSERT(cardinality_ > 0);
    const typename int_value_map_type::forward_map_type& map = valueMap.forwardMap();
    for (typename int_value_map_type::forward_map_type::const_iterator it = map.begin(); it != map.end(); it++ )
      DEBUG_BOUNDS(it->first, cardinality_);
  }

  /** Return the cardinality of this variable */
  int cardinality() const { return cardinality_; }

  /** Return the int-value map of the variable */
  int_value_map_type*  valueMap() const { return valueMap_; }

  void print1(bool detail) const
  {
    cout << "Variable " << name_;
    if (!detail) return;
    cout << " (discrete):";
    typedef typename int_value_map_type::forward_map_type map_t;
    const map_t& map = valueMap_->forwardMap();
    BOOST_FOREACH (const typename map_t::value_type& p, map)
      cout << " " << (p.first) << " <--> " << (p.second);
  }
  std::string toString(bool detail) const
  {
    std::ostringstream cout;
    cout << "Variable " << name_;
    if (!detail) return cout.str();
    cout << " (discrete):";
    typedef typename int_value_map_type::forward_map_type map_t;
    const map_t& map = valueMap_->forwardMap();
    BOOST_FOREACH (const typename map_t::value_type& p, map)
      cout << " " << (p.first) << " <--> " << (p.second);
    return cout.str();
  }
protected:
  int cardinality_;
  int_value_map_type* valueMap_;
};

END_GRAPHICAL_MODEL_NAMESPACE;
#endif // VARIABLE_H
