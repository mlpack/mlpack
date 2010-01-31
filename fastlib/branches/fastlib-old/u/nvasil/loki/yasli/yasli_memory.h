#ifndef YASLI_MEMORY_H_
#define YASLI_MEMORY_H_

// $Id: yasli_memory.h 754 2006-10-17 19:59:11Z syntheticpp $


#include "yasli_traits.h"
#include "yasli_protocols.h"//!
#include <cassert>
#include <cstddef>
#include <misc/mojo.h>//NOT A SAFE WAY TO INCLUDE IT

namespace yasli {
          
         
    // 20.4.1, the default allocator:
    template <class T> class allocator;
    template <> class allocator<void>;
    
    // 20.4.1.2, allocator globals
    template <class T, class U>
    bool operator==(const allocator<T>&, const allocator<U>&) throw()
    { return true; }

    template <class T, class U>
    bool operator!=(const allocator<T>&, const allocator<U>&) throw()
    { return false; }
    
    // 20.4.2, raw storage iterator:
    // @@@ not defined, use the std one @@@
    //template <class OutputIterator, class T> class raw_storage_iterator;
    
    // 20.4.3, temporary buffers:
    // @@@ not defined, use the std one @@@
    //template <class T>
    //pair<T*,ptrdiff_t> get_temporary_buffer(ptrdiff_t n);
    // @@@ not defined, use the std one @@@
    // template <class T>
    // void return_temporary_buffer(T* p);

    // 20.4.4, specialized algorithms:
    template <class InputIterator, class ForwardIterator>
    ForwardIterator
    uninitialized_copy(InputIterator first, InputIterator last,
    ForwardIterator result);

    template <class ForwardIterator, class Size, class T>
    void uninitialized_fill_n(ForwardIterator first, Size n, const T& x);
    // 20.4.5, pointers:
    // @@@ not defined, use the std one @@@
    // template<class X> class auto_ptr;
}

namespace yasli {
    template <class T> class allocator;
    // specialize for void:
    template <> class allocator<void> 
    {
    public:
        typedef void* pointer;
        typedef const void* const_pointer;
        // reference-to-void members are impossible.
        typedef void value_type;
        template <class U> struct rebind { typedef allocator<U> other; };
    };

    template <class T> class allocator 
    {
    public:
        typedef size_t                 size_type;
        typedef std::ptrdiff_t         difference_type;
        typedef T*                     pointer;
        typedef const T*               const_pointer;
        typedef T&                     reference;
        typedef const T&               const_reference;
        typedef T                      value_type;
        
        template <class U> struct rebind { typedef allocator<U> other; };
        allocator() throw() {}
        allocator(const allocator&) throw() {}
        template <class U> allocator(const allocator<U>&) throw() {}
        ~allocator() throw() {}
        pointer address(reference x) const { return &x; }
        const_pointer address(const_reference x) { return &x; }
        pointer allocate(size_type n, allocator<void>::const_pointer = 0)
        {
            return static_cast<pointer>(::operator new(n * sizeof(T)));
        }
        void deallocate(pointer p, size_type) 
        {
            ::operator delete(p);
        }
        size_type max_size() const throw() 
        {
            return size_type(-1);
        }
        void construct(pointer p, const T& val) 
        {
            new((void *) p) T(val);
        }
        void destroy(pointer p) 
        {
            ((T*) p)->~T();
        }
    };
} // namespace yasli

namespace yasli_nstd
{
    template <class T> class mallocator 
    {
    public:
        typedef size_t       size_type;
        typedef ptrdiff_t    difference_type;
        typedef T*           pointer;
        typedef const T*     const_pointer;
        typedef T&           reference;
        typedef const T&     const_reference;
        typedef T            value_type;
        
        template <class U> struct rebind { typedef mallocator<U> other; };
        mallocator() throw() {}
        mallocator(const mallocator&) throw() {}
        template <class U> mallocator(const mallocator<U>&) throw() {}
        ~mallocator() throw() {}
        pointer address(reference x) const { return &x; }
        const_pointer address(const_reference x) { return &x; }
        pointer allocate(size_type n, yasli::allocator<void>::const_pointer = 0)
        {
            return static_cast<pointer>(malloc(n * sizeof(T)));
        }
        void deallocate(pointer p, size_type) 
        {
            free(p);
        }
        size_type max_size() const throw() 
        {
            return size_type(-1);
        }
        void construct(pointer p, const T& val) 
        {
            new((void *) p) T(val);
        }
        void destroy(pointer p) 
        {
            ((T*) p)->~T();
        }
    };
    
    //--------------destroy--------
    
    namespace _impl
    {                  
       struct non_destroyer
       {
           template <class A, class T>   
           static void destroy(A& a, T* p, typename A::size_type n) {}
              
           template <class ForwardIterator>
           static void destroy_range(ForwardIterator b, ForwardIterator e) {} 
       };
       
       struct destroyer
       {
           template <class A, class T>
           static void destroy(A& a, T* p, typename A::size_type n)
           {
               const typename A::pointer p1 = p + n;
               for (; p < p1; ++p) a.destroy(p);
           }
              
           template <class ForwardIterator>
           static void destroy_range(ForwardIterator b, ForwardIterator e) 
           {
               typedef typename std::iterator_traits<ForwardIterator>::value_type
                  value_type;
               for (; b != e; ++b) (*b).~value_type();
           }
       };    
    }

    template <class A, class T>
    void destroy(A& a, T* p, typename A::size_type n) 
    {
        yasli_nstd::type_selector<yasli_nstd::is_class<T>::value != 0,
                                  _impl::destroyer,
                                  _impl::non_destroyer
                                 >::result::destroy(a, p, n);
    }
    
    template <class ForwardIterator>
    void destroy_range(ForwardIterator b, ForwardIterator e) 
    {
        yasli_nstd::type_selector<
            yasli_nstd::is_class<typename std::iterator_traits<ForwardIterator>
            ::value_type>::value != 0,
            _impl::destroyer,
            _impl::non_destroyer
            >::result::destroy_range(b, e);
    }
    
    //---------------


    template <class It1, class It2>
    It2 uninitialized_move(It1 b, It1 e, It2 d)
    {
        return mojo::uninitialized_move(b, e, d);
    }
    
    template <class A>
    struct generic_allocator_traits
    {
        static typename A::pointer 
        reallocate(
            A& a, 
            typename A::pointer b, 
            typename A::pointer e, 
            typename A::size_type newSize) 
        {
            typename A::pointer p1 = a.allocate(newSize, b);
            const typename A::size_type oldSize = e - b;
            if (oldSize <= newSize) // expand
            {
                yasli_protocols::move_traits<typename A::value_type>::destructive_move(
                    b, b + oldSize, p1);
            }
            else // shrink
            {
                yasli_protocols::move_traits<typename A::value_type>::destructive_move(
                    b, b + newSize, p1);
                yasli_nstd::destroy(a, b + newSize, oldSize - newSize);
            }
            a.deallocate(b, oldSize);
            return p1;
        }

        static bool reallocate_inplace(
        A& a,
        typename A::pointer b,
        typename A::size_type newSize) 
        {
            return false;
        }

    private:
        generic_allocator_traits();
    };

    template <class A>
    struct allocator_traits : public generic_allocator_traits<A> 
    {
    };

    template <class T>
    struct allocator_traits< yasli::allocator<T> >  
        : public generic_allocator_traits< yasli::allocator<T> > 
    {
#if YASLI_NEW_IS_MALLOC != 0
        
        static bool reallocate_inplace(
                        A& a,
                        typename A::pointer b,
                        typename A::size_type newSize) 
        {
            allocator_traits< yasli_nstd::mallocator<T> >
                              ::reallocate_inplace(a, b, newSize);
        }
               
        static typename yasli::allocator<T>::pointer 
        reallocate(
            yasli::allocator<T>& a, 
            typename yasli::allocator<T>::pointer b, 
            typename yasli::allocator<T>::pointer e, 
            typename yasli::allocator<T>::size_type newSize) 
        {    
            allocator_traits< yasli_nstd::mallocator<T> >
                              ::reallocate(a, b, e, newSize);      
        }
#endif//yasli_new_is_malloc
    };

    template <class T>
    struct allocator_traits< yasli_nstd::mallocator<T> >  
        : public generic_allocator_traits< yasli_nstd::mallocator<T> > 
    {
#if YASLI_HAS_EXPAND && YASLI_HAS_EFFICIENT_MSIZE
        static bool reallocate_inplace(
                        yasli_nstd::mallocator<T>& a,
                        typename yasli_nstd::mallocator<T>::pointer b,
                        typename yasli_nstd::mallocator<T>::size_type newSize) 
        {
            if (b == 0) return malloc(newSize);
            if (newSize == 0) {free(b); return false;}
            return b == yasli_platform::expand(b, newSize) 
                   && yasli_platform::msize(b) >= newSize;
        } 
#endif
        static typename yasli_nstd::mallocator<T>::pointer 
        reallocate(
            yasli_nstd::mallocator<T>& a,
            typename yasli_nstd::mallocator<T>::pointer b,
            typename yasli_nstd::mallocator<T>::pointer e,
            typename yasli_nstd::mallocator<T>::size_type newSize)
        {
            if (yasli_nstd::is_memmoveable<T>::value)
            {
                return static_cast<T*>(realloc(b, newSize));
            }
            if(reallocate_inplace(a, b, newSize)) return b;           
            return generic_allocator_traits< yasli_nstd::mallocator<T> >::
                          reallocate(a, b, e, newSize);            
        }
    };
}

namespace yasli
{
     //Here is where type_selector is really much more ugly than 
     //enable_if.
          
    //----------------UNINIT COPY--------
    namespace _impl
    {                   
          //safe
          template <class InputItr, class FwdItr>
          struct uninitialized_safe_copier
          {
             static FwdItr execute(InputItr first, InputItr last, FwdItr result)
             {
                 //
                 struct ScopeGuard
                 {
                     FwdItr begin;
                     FwdItr* current;
                     ~ScopeGuard()
                     {
                         if (!current) return;
                         FwdItr end = *current;
                         typedef typename std::iterator_traits<FwdItr>::value_type T;
                         for (; begin != end; ++begin) (&*begin)->~T();
                     }
                 } guard = { result, &result };
                 for (; first != last; ++first, ++result) 
                     new(&*result) typename std::iterator_traits<FwdItr>::value_type(*first);
                 // commit
                 return result;
             }
          };                    
          
          template <class T>
          struct uninitialized_memcopier
          {
             static T* execute(const T* first, const T* last, T* result)
             {
                 yasli_nstd::is_memcopyable<T>::value;
                 const size_t s = last - first;
                 memmove(result, first, s * sizeof(T));
                 return result + s;                 
             }
          };
            
    }// _impl
    
    // @@@ TODO: specialize for yasli_nstd::fill_iterator 
   
    template <class InputItr, class FwdItr>
    FwdItr uninitialized_copy(InputItr first, InputItr last, FwdItr result)
    {
           std::cout<<"neither\n";
        return _impl::uninitialized_safe_copier<InputItr, FwdItr>::execute(first, last, result);
    }
    
    template <class T>
    T* uninitialized_copy(const T* first, const T* last, T* result)
    {
       std::cout<<"const\n";
       return yasli_nstd::type_selector<yasli_nstd::is_memcopyable<T>::value != 0,
                                         _impl::uninitialized_memcopier<T>,
                                         _impl::uninitialized_safe_copier<const T*, T*>
                                        >::result::execute(first, last, result);
    }
    
    template <class T>
    T* uninitialized_copy(T* first, T* last, T* result)
    {
       std::cout<<"non-const\n";
       return uninitialized_copy(static_cast<const T*>(first), 
                                 static_cast<const T*>(last), result);
    }  
   
    //-------------------------UNINIT FILL------
    
    template <class ForwardIterator, class T>
    void
    uninitialized_fill(ForwardIterator first, ForwardIterator last,
        const T& x)
    {
        struct ScopeGuard
        {
            ForwardIterator first;
            ForwardIterator* pCrt;
            ~ScopeGuard()
            {
                if (pCrt) yasli_nstd::destroy_range(first, *pCrt);
            }
        } guard = { first, &first };
        for (; first != last; ++first)
            new(&*first) T(x); 
        // Commit
        guard.pCrt = 0;
    }

    template <class T, class U>
    void
    uninitialized_fill(T* first, T* last, const U& x)
    {
        struct ScopeGuard
        {
            T* first;
            T** pCrt;
            ~ScopeGuard()
            {
                if (pCrt) yasli_nstd::destroy_range(first, *pCrt);
            }
        } guard = { first, &first };
        assert(first <= last);
        switch ((last - first) & 7u)
        {
        case 0:
            while (first != last)
            {
                new(first) T(x); ++first;
        case 7: new(first) T(x); ++first;
        case 6: new(first) T(x); ++first;
        case 5: new(first) T(x); ++first;
        case 4: new(first) T(x); ++first;
        case 3: new(first) T(x); ++first;
        case 2: new(first) T(x); ++first;
        case 1: new(first) T(x); ++first;
                assert(first <= last);
            }
        }
        // Commit
        guard.pCrt = 0;
    }
    
}// yasli

#endif // YASLI_MEMORY_H_
