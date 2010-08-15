#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <map>
#include <vector>
#include <set>
#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** Augment the std::set with two operations operator<< and contains() */
template<typename _Key, typename _Compare = std::less<_Key>, typename _Alloc = std::allocator<_Key> >
class Set : public std::set<_Key, _Compare, _Alloc>
{
public:
  typedef typename std::set<_Key, _Compare, _Alloc>::value_type value_type;

  Set& operator << (const value_type& x) { this->insert(x); return *this; }
  bool contains(const value_type& x) const { return this->find(x) != this->end(); }
};

/** Augment the std::vector with operator<< */
template<typename _Tp, typename _Alloc = std::allocator<_Tp> >
class Vector : public std::vector<_Tp, _Alloc>
{
public:
  typedef typename std::vector<_Tp, _Alloc>::value_type         value_type;
  typedef typename std::vector<_Tp, _Alloc>::size_type          size_type;
  typedef typename std::vector<_Tp, _Alloc>::allocator_type     allocator_type;
public:
  Vector () : std::vector<_Tp, _Alloc>() {}
  Vector(const allocator_type& __a) : std::vector<_Tp, _Alloc>(__a) {}
  Vector(size_type __n, const value_type& __value = value_type(),
     const allocator_type& __a = allocator_type()) : std::vector<_Tp, _Alloc>(__n, __value, __a) {}

  Vector& operator << (const value_type& x) { this->push_back(x); return *this; }
  void fill(const value_type& x)
  {
    for (unsigned int i = 0; i < this->size(); i++) (*this)[i] = x;
  }
};

/** Augment the std::Map with contains() */
template <typename _Key, typename _Tp, typename _Compare = std::less<_Key>,
          typename _Alloc = std::allocator<std::pair<const _Key, _Tp> > >
class Map : public std::map<_Key, _Tp, _Compare, _Alloc>
{
public:
  typedef typename std::map<_Key, _Tp, _Compare, _Alloc>::key_type     key_type;
  typedef typename std::map<_Key, _Tp, _Compare, _Alloc>::value_type   value_type;
  typedef typename std::map<_Key, _Tp, _Compare, _Alloc>::mapped_type  mapped_type;

  bool contains(const key_type& x) const { return this->find(x) != this->end(); }
};

/** The dual map for two types with comparison capability */
template <typename A, typename B> class DualMap
{
public:
  typedef std::pair<A, B>       pair_type;
  typedef std::pair<A, B>       reverse_pair_type;
  typedef Map<A, B>             forward_map_type;
  typedef Map<B, A>             reverse_map_type;
public:
  void set(const A& a, const B& b) { mapA[a] = b; mapB[b] = a; }
  bool containsForward(const A& a) const { return mapA.contains(a); }
  bool containsReverse(const B& b) const { return mapB.contains(b); }
  B getForward(const A& a) const
  {
    typename forward_map_type::const_iterator it = mapA.find(a);
    DEBUG_ASSERT(it != mapA.end());
    return (it->second);
  }
  A getReverse(const B& b) const
  {
    typename reverse_map_type::const_iterator it = mapB.find(b);
    DEBUG_ASSERT(it != mapB.end());
    return (it->second);
  }
  int size() const { return mapA.size(); }
  DualMap& operator << (const pair_type& p)
  {
    this->set(p.first, p.second);
    return *this;
  }
  const forward_map_type& forwardMap() const { return mapA; }
  const reverse_map_type& reverseMap() const { return mapB; }
protected:
  Map<A, B> mapA;
  Map<B, A> mapB;
};

END_GRAPHICAL_MODEL_NAMESPACE;

#endif // COMMON_TYPES_H
