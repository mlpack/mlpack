#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <map>
#include <vector>
#include <set>
#include <list>
#include <queue>
#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** Augment the std::set with two operations operator<< and contains() */
template<typename _Key, typename _Compare = std::less<_Key>, typename _Alloc = std::allocator<_Key> >
class Set : public std::set<_Key, _Compare, _Alloc>
{
public:
  typedef std::set<_Key, _Compare, _Alloc>    _Base;
  typedef typename _Base::value_type          value_type;

  Set& operator << (const value_type& x) { this->insert(x); return *this; }
  bool contains(const value_type& x) const { return this->find(x) != this->end(); }
};

/** Augment the std::vector with operator<< */
template<typename _Tp, typename _Alloc = std::allocator<_Tp> >
class Vector : public std::vector<_Tp, _Alloc>
{
public:
  typedef std::vector<_Tp, _Alloc>           _Base;
  typedef typename _Base::value_type         value_type;
  typedef typename _Base::size_type          size_type;
  typedef typename _Base::allocator_type     allocator_type;
public:
  Vector () : _Base() {}
  Vector(const allocator_type& __a) : _Base(__a) {}
  Vector(size_type __n, const value_type& __value = value_type(),
     const allocator_type& __a = allocator_type()) : _Base(__n, __value, __a) {}

  Vector& operator << (const value_type& x) { this->push_back(x); return *this; }
  void fill(const value_type& x)
  {
    for (unsigned int i = 0; i < this->size(); i++) (*this)[i] = x;
  }
};

/** Augment the std::list with operator<< */
template<typename _Tp, typename _Alloc = std::allocator<_Tp> >
class List : public std::list<_Tp, _Alloc>
{
public:
  typedef std::list<_Tp, _Alloc>             _Base;
  typedef typename _Base::value_type         value_type;
  typedef typename _Base::size_type          size_type;
  typedef typename _Base::allocator_type     allocator_type;
public:
  List () : _Base() {}
  List(const allocator_type& __a) : _Base(__a) {}
  List(size_type __n, const value_type& __value = value_type(),
     const allocator_type& __a = allocator_type()) : _Base(__n, __value, __a) {}

  List& operator << (const value_type& x) { this->push_back(x); return *this; }
};

/** Augment the std::Map with contains(), get and operator << */
template <typename _Key, typename _Tp, typename _Compare = std::less<_Key>,
          typename _Alloc = std::allocator<std::pair<const _Key, _Tp> > >
class Map : public std::map<_Key, _Tp, _Compare, _Alloc>
{
public:
  typedef std::map<_Key, _Tp, _Compare, _Alloc>     _Base;
  typedef typename _Base::key_type                  key_type;
  typedef typename _Base::value_type                value_type;
  typedef typename _Base::mapped_type               mapped_type;
  typedef typename _Base::const_iterator            const_iterator;

  bool contains(const key_type& x) const { return this->find(x) != this->end(); }
  const mapped_type& get(const key_type& x) const
  {
    const_iterator it = this->find(x);
    DEBUG_ASSERT(it != this->end());
    return it->second;
  }
  Map& operator<< (const value_type& p)
  {
    (*this)[p.first] = p.second;
    return *this;
  }
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

/** Augment the priority queue with operator<<*/
template<typename _Tp, typename _Sequence = std::vector<_Tp>, typename _Compare  = std::less<typename _Sequence::value_type> >
class PriorityQueue : public std::priority_queue<_Tp, _Sequence, _Compare>
{
  typedef std::priority_queue<_Tp, _Sequence, _Compare>            _Base;
  typedef typename _Base::value_type                               value_type;
public:
  PriorityQueue(const _Compare& __comp = _Compare(), const _Sequence& __s = _Sequence()) : _Base(__comp, __s) {}
  PriorityQueue& operator << (const value_type& x) { this->push(x); return *this; }
};

END_GRAPHICAL_MODEL_NAMESPACE;

#endif // COMMON_TYPES_H
