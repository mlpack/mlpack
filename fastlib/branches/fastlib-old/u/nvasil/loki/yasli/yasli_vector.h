#ifndef YASLI_VECTOR_H_
#define YASLI_VECTOR_H_

// $Id: yasli_vector.h 754 2006-10-17 19:59:11Z syntheticpp $


#include "platform.h"
#include "yasli_fill_iterator.h"
#include "yasli_memory.h"
#include "yasli_traits.h"
#include "yasli_protocols.h"
#include <iterator>
#include <cassert>
#include <stdexcept>
#include "../TypeManip.h"

namespace yasli 
{
    template <class T, class Allocator = allocator<T> >
    class vector;
}

namespace yasli {
     
   template <class T, class Allocator>
   class vector 
   {
       struct ebo : public Allocator
       {
           T *beg_;
           ebo() {}
           ebo(const Allocator& a) : Allocator(a) {}
       } ebo_;
       T *end_;
       T *eos_; 
   public:
       // types:
       typedef          vector<T, Allocator>             this_type;//not standard
       typedef typename Allocator::reference             reference;
       typedef typename Allocator::const_reference       const_reference;
       typedef typename Allocator::pointer               iterator;       // See 23.1
       typedef typename Allocator::const_pointer         const_iterator; // See 23.1
       typedef typename Allocator::size_type             size_type;      // See 23.1
       typedef typename Allocator::difference_type       difference_type;// See 23.1
       typedef          T                                value_type;
       typedef          Allocator                        allocator_type;
       typedef typename Allocator::pointer               pointer;
       typedef typename Allocator::const_pointer         const_pointer;
       typedef std::reverse_iterator<iterator>           reverse_iterator;
       typedef std::reverse_iterator<const_iterator>     const_reverse_iterator;
   private:
       void init_empty() 
       {
            
   #if YASLI_UNDEFINED_POINTERS_COPYABLE == 1
           end_ = ebo_.beg_;
           eos_ = ebo_.beg_;
   #else   
           ebo_.beg_ = 0;
           end_ = 0;
           eos_ = 0;    
   #endif
           assert(empty());
           
       }
       
       void init_move(vector& temp)
       {
           ebo_ = temp.ebo_;
           end_ = temp.end_;
           eos_ = temp.eos_;
           temp.init_empty();
       }
   
       void init_fill(size_type n, const T& value, const Allocator& a)
       {
           // Will avoid catch (...)
           vector<T, Allocator> temp(a);
           temp.insert(temp.end(), n, value);
           init_move(temp);
           assert(size() == n);
       }
       
       // 23.2.4.1 construct/copy/destroy:
       template <class InputIterator, class looks_like_itr>
       void init(InputIterator first, InputIterator last, looks_like_itr, const Allocator& a)
       {
           vector temp(a);
           temp.insert(temp.end(), first, last);
           init_move(temp);
       }
       
       template <class non_iterator>
       void init(non_iterator n, non_iterator datum, Loki::Int2Type<false>, 
                                                     const Allocator& a)
       {
            init_fill((size_type)n, (const T&)datum, a);
       }
       
   public:
       vector()
       {
           init_empty();
       }
       
       explicit vector(const Allocator& a) 
           : ebo_(a)
       {
           init_empty();
       }
   
       explicit vector(size_type n, const T& value = T(),
               const Allocator& a = Allocator())
       {
           init_fill(n, value, a);
       }
       
       //!! avoid enable_if
       template <class InputIterator>
       vector(InputIterator first, InputIterator last, const Allocator& a = Allocator())
       {                
            init(first, last, Loki::Int2Type<
                       yasli_nstd::is_class<InputIterator>::value || 
                       yasli_nstd::is_pointer<InputIterator>::value >(), a);
       }
       
   public:
       vector(const vector<T,Allocator>& x)
       {
           vector temp(x.begin(), x.end(), x.ebo_);
           init_move(temp);
       }
   
       ~vector()
       {
           yasli_nstd::destroy(ebo_, ebo_.beg_, size());
           const size_type c = capacity();
           if (c != 0) ebo_.deallocate(ebo_.beg_, c);
       }
   
       // Note pass by value
       vector<T,Allocator>& operator=(vector<T,Allocator> temp)
       {
           temp.swap(*this);
           return *this;
       }
   
       template <class InputIterator>   
       void assign(InputIterator first, InputIterator last)
       {
                     
           assign_pre_impl(first, last, Loki::Int2Type<yasli_nstd::
                                  is_class<InputIterator>::value||yasli_nstd::
                                  is_pointer<InputIterator>::value>());
       }
       
   private://-------ASSIGN IMPLEMENTATION
       template <class InputIterator, class looks_like_itr>
       void assign_pre_impl(InputIterator first, InputIterator last, looks_like_itr)
       {    
            assign_impl(first, last,
               std::iterator_traits<InputIterator>::iterator_category());
       }
       
       template <class InputIterator>
       void assign_pre_impl(InputIterator n, InputIterator datum, Loki::Int2Type<false>)
       {    
            assign((size_type) n, (const T&) datum);
       }
   
       template <class InputIterator>
       void assign_impl(InputIterator first, InputIterator last, std::input_iterator_tag)
       {
           for (iterator i = begin(); i != end(); ++i, ++first)
           {
               if (first == last) 
               {
                   resize(i - begin());
                   return;
               }
               *i = *first;
           }
           // we filled up the vector, now insert the rest
           insert(end(), first, last);
       }
   
       template <class RanIt>
       void assign_impl(RanIt first, RanIt last, std::random_access_iterator_tag)
       {
           const typename std::iterator_traits<RanIt>::difference_type d = 
               last - first;
           assert(d >= 0);
           size_type newSize = size_type(d);
           assert(newSize == d); // no funky iterators
           reserve(newSize);
           if (newSize >= size())
           {
               const size_t delta = newSize - size();
               RanIt i = last - delta;
               copy(first, i, ebo_.beg_);
               insert(end(), i, last);
           }
           else
           {
               copy(first, last, ebo_.beg_);
               resize(newSize);
           }
           assert(size() == newSize);
       }
   public:
       void assign(size_type n, const T& u)
       {
           const size_type s = size();
           if (n <= s)
           {
               T* const newEnd = ebo_.beg_ + n;
               fill(ebo_.beg_, newEnd, u);
               yasli_nstd::destroy(ebo_, newEnd, s - n);
               end_ = newEnd;
           }
           else
           {
               reserve(n);
               T* const newEnd = ebo_.beg_ + n;
               fill(ebo_.beg_, end_, u);
               uninitialized_fill(end_, newEnd, u);
               end_ = newEnd;
           }
           assert(size() == n);
           assert(empty() || front() == back());
       }
   
       allocator_type get_allocator() const
       {
           return ebo_;
       }
       // iterators:
       iterator begin() { return ebo_.beg_; }
       const_iterator begin() const { return ebo_.beg_; }
       iterator end() { return end_; }
       const_iterator end() const { return end_; }
       reverse_iterator rbegin() { return reverse_iterator(end()); }
       const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
       reverse_iterator rend() { return reverse_iterator(begin()); }
       const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
   
       // 23.2.4.2 capacity:
   
       size_type size() const { return end_ - ebo_.beg_; }
       size_type max_size() const { return ebo_.max_size(); }
   
       void resize(size_type sz, T c = T()) 
       {
           const size_type oldSz = size();
           if (oldSz >= sz)
           {
               erase(ebo_.beg_ + sz, end_);
           }
           else
           {
               reserve(sz);
               uninitialized_fill(end_, end_ + (sz - oldSz), c);
               end_ = ebo_.beg_ + sz;
           }
           assert(size() == sz);
       }
   private:
       template<class is>
       void resize_impl(size_type sz, T c)
       {
       }
   public:
       size_type capacity() const
       {
           return eos_ - ebo_.beg_; 
       }
       bool empty() const
       {
           return ebo_.beg_ == end_; 
       }
       void reserve(size_type n)
       {
           const size_type 
               s = size(),
               c = capacity();
           if (c >= n) return;
           if (capacity() == 0)
           {
               ebo_.beg_ = ebo_.allocate(n);
           }
           else
           {
               ebo_.beg_ = yasli_nstd::allocator_traits<Allocator>::reallocate(
                   ebo_, ebo_.beg_, end_, n);
           }
           end_ = ebo_.beg_ + s;
           eos_ = ebo_.beg_ + n;
           assert(capacity() >= n);
           assert(size() == s);
       }
       bool reserve_inplace_nstd(size_type n)
       {
           if (capacity() >= n) return true;
           if (!yasli_nstd::allocator_traits<Allocator>::reallocate_inplace(
               ebo_, ebo_.beg_, n)) 
           {
               return false;
           }
           eos_ = ebo_.beg_ + n;
           return true;
       }
       // element access:
       reference operator[](size_type n)
       {
           assert(n < size());
           return ebo_.beg_[n];
       }
       const_reference operator[](size_type n) const
       {
           assert(n < size());
           return ebo_.beg_[n];
       }
       const_reference at(size_type n) const
       {
           // Fix by Joseph Canedo
           if (n >= size()) throw std::range_error("vector<>::at");
           return ebo_.beg_[n];
       }
       reference at(size_type n)
       {
           // Fix by Joseph Canedo
           if (n >= size()) throw std::range_error("vector<>::at");
           return ebo_.beg_[n];
       }
       reference front()
       {
           assert(!empty());
           return *ebo_.beg_;
       }
       const_reference front() const
       {
           assert(!empty());
           return *ebo_.beg_;
       }
       reference back()
       {
           assert(!empty());
           return end_[-1];
       }
       const_reference back() const
       {
           assert(!empty());
           return end_[-1];
       }
   
   private:
           
       void prepare_growth(size_type delta)
       {
           const size_type s = size();
           // @@@ todo: replace magic constant with something meaningful
           const size_type smallThreshold = 8;
           if (s < smallThreshold)
           {
               reserve(std::max(smallThreshold, delta));
           }
           else
           {
               const size_type multiply = 3;
               const size_type divide = 2;
               const size_type suggestedSize = (s * multiply) / divide;
               reserve(std::max(s + delta, suggestedSize));
           }
       }
   
   public:
       // 23.2.4.3 modifiers:
       void push_back(const T& x)
       {
           if (size() == capacity()) 
           {
               prepare_growth(1);
           }
           new(end_) T(x);
           ++end_;
       }
       void pop_back()
       {
           assert(!empty());
           ebo_.destroy(--end_);
       }
       void move_back_nstd(T& x)
       {
           if (size() == capacity()) 
           {
               prepare_growth(1);
           }
           yasli_protocols::move_traits<T>::nondestructive_move(&x, &x + 1, end_);
       }
   
       // 23.2.4.3 modifiers:
       iterator insert(iterator position, const T& x)
       {
           // @@@ be smarter about this reservation 
           reserve(size() + 1);
           const size_type pos = position - begin();
           insert(position, (size_type)1, x);
           return ebo_.beg_ + pos;
       }
       void insert(iterator position, size_type n, const T& x)
       {
           insert(position, 
               yasli_nstd::fill_iterator<const T&>(x),
               yasli_nstd::fill_iterator<const T&>(x, n)
               );
       }
       
       template <class InputIterator>
       void insert(iterator position, InputIterator first, InputIterator last)
       {
           insert_pre_impl(position, first, last,
              Loki::Int2Type<yasli_nstd::is_class<InputIterator>::value||
                             yasli_nstd::is_pointer<InputIterator>::value>()); 
       }
   private:
       template<class InputIterator, class looks_like_iterator>
       void 
       insert_pre_impl(iterator position, InputIterator first, InputIterator last,
                       looks_like_iterator)
       {      
           insert_impl(position, first, last, 
           typename std::iterator_traits<InputIterator>::iterator_category());
       }
       
       template <class non_iterator>
       void insert_pre_impl(iterator position, non_iterator n, non_iterator x, 
                                                         Loki::Int2Type<false>)
       {   //used if e.g. T is int and insert(itr, 10, 6) is called
           insert(position, static_cast<size_type>(n), 
                            static_cast<value_type>(x));
       }
       
       template <class InputIterator>
       void insert_impl(iterator position,
           InputIterator first, InputIterator last, std::input_iterator_tag)
       {
           for (; first != last; ++first)
           {
               position = insert(position, *first) + 1;
           }
       }
       template <class FwdIterator>
       void insert_impl(iterator position,
           FwdIterator first, FwdIterator last, std::forward_iterator_tag)
       {
           typedef yasli_protocols::move_traits<T> mt;
           
           const typename std::iterator_traits<FwdIterator>::difference_type 
                          count = std::distance(first, last);
               
           if (eos_ - end_ > count || reserve_inplace_nstd(size() + count)) // there's enough room
           {
               if (count > end_ - &*position)
               {
                   // Step 1: fill the hole between end_ and position+count
                   FwdIterator i1 = first; 
                   std::advance(i1, end_ - &*position);
                   FwdIterator i2 = i1;
                   std::advance(i2, &*position + count - end_);//why not i2 = first; advance(i2,count);
                   T* const oldEnd = end_;
                   end_ = copy(i1, i2, end_);
                   assert(end_ == &*position + count);
                   // Step 2: move existing data to the end
                   mt::nondestructive_move(
                       position,
                       oldEnd,
                       end_);
                   end_ = oldEnd + count;
                   // Step 3: copy in the remaining data
                   copy(first, i1, position);
               }
               else // simpler case
               {
                   mt::nondestructive_move(
                       end_ - count,
                       end_,
                       end_);
                   end_ += count;
                   mt::nondestructive_assign_move(
                       position,
                       end_ - count,
                       position + count);
                   copy(first, last, position);
               }
           }
           else
           {
               vector<T, Allocator> temp(ebo_);
               temp.reserve(size() + count);
               // The calls below won't cause infinite recursion
               //   because they will fall on the other branch
               //   of the if statement
               temp.insert(temp.end(), begin(), position);
               temp.insert(temp.end(), first, last);
               temp.insert(temp.end(), position, end());
               assert(temp.size() == size() + count);
               temp.swap(*this);
           }
       }
   public:
   
       iterator erase(iterator position)
       {
           erase(position, position + 1);
           return position;
       }
       iterator erase(iterator first, iterator last)
       {
           yasli_protocols::move_traits<T>::nondestructive_assign_move(
               last, end(), first);
           Allocator& a = ebo_;
           const size_type destroyed = last - first;
           yasli_nstd::destroy(a, end_ - destroyed, destroyed);
           end_ -= destroyed;
           return first;
       }
       void swap(vector<T,Allocator>& rhs)//COULD DO THIS WITH LESS TEMPORARIES
       {
           std::swap(static_cast<Allocator&>(ebo_), static_cast<Allocator&>(rhs.ebo_));
           std::swap(ebo_.beg_, rhs.ebo_.beg_);
           std::swap(end_, rhs.end_);
           std::swap(eos_, rhs.eos_);
       }
       void clear()
       {
           Allocator& a = ebo_;
           yasli_nstd::destroy(a, ebo_.beg_, size());
           end_ = ebo_.beg_;
       }
   };//vector
   
   
   
   template <class T, class Allocator>
   bool operator==(const vector<T,Allocator>& x,
                   const vector<T,Allocator>& y);
   template <class T, class Allocator>
   bool operator< (const vector<T,Allocator>& x,
                   const vector<T,Allocator>& y);
   template <class T, class Allocator>
   bool operator!=(const vector<T,Allocator>& x,
                   const vector<T,Allocator>& y);
   template <class T, class Allocator>
   bool operator> (const vector<T,Allocator>& x,
                   const vector<T,Allocator>& y);
   template <class T, class Allocator>
   bool operator>=(const vector<T,Allocator>& x,
                   const vector<T,Allocator>& y);
   template <class T, class Allocator>
   bool operator<=(const vector<T,Allocator>& x,
                   const vector<T,Allocator>& y);
   // specialized algorithms:
   template <class T, class Allocator>
   void swap(vector<T,Allocator>& x, vector<T,Allocator>& y);
   
}//yasli



namespace yasli_protocols
{         
    template <class T, class A>
    struct move_traits< yasli::vector<T, A> >:public 
    yasli_nstd::type_selector<
                    sizeof(yasli::vector<T, A>) != (3 * sizeof(T*)),
                    memmove_traits< std::complex<T> >,
                    safe_move_traits< std::complex<T> >
             >::result 
    {
    };
}

#endif