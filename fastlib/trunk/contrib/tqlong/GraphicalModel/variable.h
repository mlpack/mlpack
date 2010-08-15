#ifndef VARIABLE_H
#define VARIABLE_H

#include "gm.h"
#include <iostream>
using namespace std;
BEGIN_GRAPHICAL_MODEL_NAMESPACE;
/**
  * A basic class represents a random variable
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
  int type() const { return type_; }
  const std::string& name() const { return name_; }
  virtual int cardinality() const = 0;
  virtual void print() const
  {
    cout << "name = " << name_ << " type = " << type_;
  }
};

/**
  * A discrete random variable implementation
  * a variable can only be assigned value from a finite set of values. Usage:

  //First define variable templates
  FiniteVar<int> tmp1("tmp1", 2);
  FiniteVar<string> tmp2("tmp2", valueMap); // DualMap<int, string> valueMap;
  //then add new variables to the universe using templates
  FiniteVariable *x = u.newFiniteVariable("x", tmp1),
                 *y = u.newFiniteVariable("y", tmp1),
                 *z = u.newFiniteVariable("z", tmp2);

  x, y, z will have same cardinality_, values_ and valueMap_ as tmp1, tmp2
  while having their own names.
  */
template <typename _V>  class FiniteVar : public Variable
{
public:
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

  void print() const
  {
    cout << "name = " << name_ << " (discrete):";
    typedef typename int_value_map_type::forward_map_type map_t;
    const map_t& map = valueMap_->forwardMap();
    BOOST_FOREACH (const typename map_t::value_type& p, map)
      cout << " " << (p.first) << " <--> " << (p.second);
  }
protected:
  int cardinality_;
  int_value_map_type* valueMap_;
};

//class FiniteVariable : public Variable
//{
//public:
//  typedef Set<Value, ValueCompare> value_set_type;
//  typedef DualMap<int, std::string> value_string_map_type;
//public:
//  /** Create a finite variable with its cardinality */
//  FiniteVariable(const std::string& name, int cardinality)
//    : Variable(name, 0), cardinality_(cardinality), valueMap_(NULL)
//  {
//    DEBUG_ASSERT(cardinality > 0);
//    values_ = new value_set_type;
//    for (int val = 0; val < cardinality; val++) (*values_) << val;
//  }
//
////  FiniteVariable(const std::string& name, value_set_type& values)
////    : Variable(name, 0), cardinality_(values.size()), values_(&values), valueMap_(NULL)
////  {
////  }
//
//  /** Create a finite variable with values being strings, they are converted
//    * to int
//    */
//  FiniteVariable(const std::string& name, value_string_map_type& valueMap)
//    : valueMap_(&valueMap)
//  {
//    cardinality_ = valueMap.size();
//    DEBUG_ASSERT(cardinality_ > 0);
//    values_ = new value_set_type;
//    for (int val = 0; val < cardinality_; val++) (*values_) << val;
//  }
//
//  /** Copy constructor */
//  FiniteVariable(const std::string &name, const FiniteVariable& t1)
//    : Variable(name, 0), cardinality_(t1.cardinality_), values_(t1.values_), valueMap_(t1.valueMap_)
//  {
//  }
//
//  int cardinality() const { return cardinality_; }
//  value_set_type* values() const { return values_; }
//  value_string_map_type* valueMap() const { return valueMap_; }
//protected:
//  int cardinality_;
//  value_set_type* values_;
//  value_string_map_type* valueMap_;
//};

END_GRAPHICAL_MODEL_NAMESPACE;
#endif // VARIABLE_H
