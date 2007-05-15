////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2001 by Andrei Alexandrescu
// This code accompanies the book:
// Alexandrescu, Andrei. "Modern C++ Design: Generic Programming and Design 
//     Patterns Applied". Copyright (c) 2001. Addison-Wesley.
// Permission to use, copy, modify, distribute and sell this software for any 
//     purpose is hereby granted without fee, provided that the above copyright 
//     notice appear in all copies and that both that copyright notice and this 
//     permission notice appear in supporting documentation.
// The author or Addison-Welsey Longman make no representations about the 
//     suitability of this software for any purpose. It is provided "as is" 
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////
#ifndef LOKI_TYPELIST_INC_
#define LOKI_TYPELIST_INC_

// $Id: Typelist.h 749 2006-10-17 19:49:26Z syntheticpp $


#include "NullType.h"
#include "TypeManip.h"
#include "TypelistMacros.h"


namespace Loki
{
////////////////////////////////////////////////////////////////////////////////
// class template Typelist
// The building block of typelists of any length
// Use it through the LOKI_TYPELIST_NN macros
// Defines nested types:
//     Head (first element, a non-typelist type by convention)
//     Tail (second element, can be another typelist)
////////////////////////////////////////////////////////////////////////////////

    template <class T, class U>
    struct Typelist
    {
       typedef T Head;
       typedef U Tail;
    };

// Typelist utility algorithms

    namespace TL
    {

////////////////////////////////////////////////////////////////////////////////
// class template MakeTypelist
// Takes a number of arguments equal to its numeric suffix
// The arguments are type names.
// MakeTypelist<T1, T2, ...>::Result
// returns a typelist that is of T1, T2, ...
////////////////////////////////////////////////////////////////////////////////

        template
        <
            typename T1  = NullType, typename T2  = NullType, typename T3  = NullType,
            typename T4  = NullType, typename T5  = NullType, typename T6  = NullType,
            typename T7  = NullType, typename T8  = NullType, typename T9  = NullType,
            typename T10 = NullType, typename T11 = NullType, typename T12 = NullType,
            typename T13 = NullType, typename T14 = NullType, typename T15 = NullType,
            typename T16 = NullType, typename T17 = NullType, typename T18 = NullType
        > 
        struct MakeTypelist
        {
        private:
            typedef typename MakeTypelist
            <
                T2 , T3 , T4 , 
                T5 , T6 , T7 , 
                T8 , T9 , T10, 
                T11, T12, T13,
                T14, T15, T16, 
                T17, T18
            >
            ::Result TailResult;

        public:
            typedef Typelist<T1, TailResult> Result;
        };

        template<>
        struct MakeTypelist<>
        {
            typedef NullType Result;
        };

////////////////////////////////////////////////////////////////////////////////
// class template Length
// Computes the length of a typelist
// Invocation (TList is a typelist):
// Length<TList>::value
// returns a compile-time constant containing the length of TList, not counting
//     the end terminator (which by convention is NullType)
////////////////////////////////////////////////////////////////////////////////

        template <class TList> struct Length;
        template <> struct Length<NullType>
        {
            enum { value = 0 };
        };
        
        template <class T, class U>
        struct Length< Typelist<T, U> >
        {
            enum { value = 1 + Length<U>::value };
        };

////////////////////////////////////////////////////////////////////////////////
// class template TypeAt
// Finds the type at a given index in a typelist
// Invocation (TList is a typelist and index is a compile-time integral 
//     constant):
// TypeAt<TList, index>::Result
// returns the type in position 'index' in TList
// If you pass an out-of-bounds index, the result is a compile-time error
////////////////////////////////////////////////////////////////////////////////

        template <class TList, unsigned int index> struct TypeAt;
        
        template <class Head, class Tail>
        struct TypeAt<Typelist<Head, Tail>, 0>
        {
            typedef Head Result;
        };

        template <class Head, class Tail, unsigned int i>
        struct TypeAt<Typelist<Head, Tail>, i>
        {
            typedef typename TypeAt<Tail, i - 1>::Result Result;
        };

////////////////////////////////////////////////////////////////////////////////
// class template TypeAtNonStrict
// Finds the type at a given index in a typelist
// Invocations (TList is a typelist and index is a compile-time integral 
//     constant):
// a) TypeAt<TList, index>::Result
// returns the type in position 'index' in TList, or NullType if index is 
//     out-of-bounds
// b) TypeAt<TList, index, D>::Result
// returns the type in position 'index' in TList, or D if index is out-of-bounds
////////////////////////////////////////////////////////////////////////////////

        template <class TList, unsigned int index,
            typename DefaultType = NullType>
        struct TypeAtNonStrict
        {
            typedef DefaultType Result;
        };
        
        template <class Head, class Tail, typename DefaultType>
        struct TypeAtNonStrict<Typelist<Head, Tail>, 0, DefaultType>
        {
            typedef Head Result;
        };
        
        template <class Head, class Tail, unsigned int i, typename DefaultType>
        struct TypeAtNonStrict<Typelist<Head, Tail>, i, DefaultType>
        {
            typedef typename 
                TypeAtNonStrict<Tail, i - 1, DefaultType>::Result Result;
        };

////////////////////////////////////////////////////////////////////////////////
// class template IndexOf
// Finds the index of a type in a typelist
// Invocation (TList is a typelist and T is a type):
// IndexOf<TList, T>::value
// returns the position of T in TList, or NullType if T is not found in TList
////////////////////////////////////////////////////////////////////////////////

        template <class TList, class T> struct IndexOf;
        
        template <class T>
        struct IndexOf<NullType, T>
        {
            enum { value = -1 };
        };
        
        template <class T, class Tail>
        struct IndexOf<Typelist<T, Tail>, T>
        {
            enum { value = 0 };
        };
        
        template <class Head, class Tail, class T>
        struct IndexOf<Typelist<Head, Tail>, T>
        {
        private:
            enum { temp = IndexOf<Tail, T>::value };
        public:
            enum { value = (temp == -1 ? -1 : 1 + temp) };
        };

////////////////////////////////////////////////////////////////////////////////
// class template Append
// Appends a type or a typelist to another
// Invocation (TList is a typelist and T is either a type or a typelist):
// Append<TList, T>::Result
// returns a typelist that is TList followed by T and NullType-terminated
////////////////////////////////////////////////////////////////////////////////

        template <class TList, class T> struct Append;
        
        template <> struct Append<NullType, NullType>
        {
            typedef NullType Result;
        };
        
        template <class T> struct Append<NullType, T>
        {
            typedef Typelist<T,NullType> Result;
        };

        template <class Head, class Tail>
        struct Append<NullType, Typelist<Head, Tail> >
        {
            typedef Typelist<Head, Tail> Result;
        };
        
        template <class Head, class Tail, class T>
        struct Append<Typelist<Head, Tail>, T>
        {
            typedef Typelist<Head, 
                    typename Append<Tail, T>::Result>
                Result;
        };
        
////////////////////////////////////////////////////////////////////////////////
// class template Erase
// Erases the first occurence, if any, of a type in a typelist
// Invocation (TList is a typelist and T is a type):
// Erase<TList, T>::Result
// returns a typelist that is TList without the first occurence of T
////////////////////////////////////////////////////////////////////////////////

        template <class TList, class T> struct Erase;
        
        template <class T>                         // Specialization 1
        struct Erase<NullType, T>
        {
            typedef NullType Result;
        };

        template <class T, class Tail>             // Specialization 2
        struct Erase<Typelist<T, Tail>, T>
        {
            typedef Tail Result;
        };

        template <class Head, class Tail, class T> // Specialization 3
        struct Erase<Typelist<Head, Tail>, T>
        {
            typedef Typelist<Head, 
                    typename Erase<Tail, T>::Result>
                Result;
        };

////////////////////////////////////////////////////////////////////////////////
// class template EraseAll
// Erases all first occurences, if any, of a type in a typelist
// Invocation (TList is a typelist and T is a type):
// EraseAll<TList, T>::Result
// returns a typelist that is TList without any occurence of T
////////////////////////////////////////////////////////////////////////////////

        template <class TList, class T> struct EraseAll;
        template <class T>
        struct EraseAll<NullType, T>
        {
            typedef NullType Result;
        };
        template <class T, class Tail>
        struct EraseAll<Typelist<T, Tail>, T>
        {
            // Go all the way down the list removing the type
            typedef typename EraseAll<Tail, T>::Result Result;
        };
        template <class Head, class Tail, class T>
        struct EraseAll<Typelist<Head, Tail>, T>
        {
            // Go all the way down the list removing the type
            typedef Typelist<Head, 
                    typename EraseAll<Tail, T>::Result>
                Result;
        };

////////////////////////////////////////////////////////////////////////////////
// class template NoDuplicates
// Removes all duplicate types in a typelist
// Invocation (TList is a typelist):
// NoDuplicates<TList, T>::Result
////////////////////////////////////////////////////////////////////////////////

        template <class TList> struct NoDuplicates;
        
        template <> struct NoDuplicates<NullType>
        {
            typedef NullType Result;
        };

        template <class Head, class Tail>
        struct NoDuplicates< Typelist<Head, Tail> >
        {
        private:
            typedef typename NoDuplicates<Tail>::Result L1;
            typedef typename Erase<L1, Head>::Result L2;
        public:
            typedef Typelist<Head, L2> Result;
        };

////////////////////////////////////////////////////////////////////////////////
// class template Replace
// Replaces the first occurence of a type in a typelist, with another type
// Invocation (TList is a typelist, T, U are types):
// Replace<TList, T, U>::Result
// returns a typelist in which the first occurence of T is replaced with U
////////////////////////////////////////////////////////////////////////////////

        template <class TList, class T, class U> struct Replace;
        
        template <class T, class U>
        struct Replace<NullType, T, U>
        {
            typedef NullType Result;
        };

        template <class T, class Tail, class U>
        struct Replace<Typelist<T, Tail>, T, U>
        {
            typedef Typelist<U, Tail> Result;
        };

        template <class Head, class Tail, class T, class U>
        struct Replace<Typelist<Head, Tail>, T, U>
        {
            typedef Typelist<Head,
                    typename Replace<Tail, T, U>::Result>
                Result;
        };

////////////////////////////////////////////////////////////////////////////////
// class template ReplaceAll
// Replaces all occurences of a type in a typelist, with another type
// Invocation (TList is a typelist, T, U are types):
// Replace<TList, T, U>::Result
// returns a typelist in which all occurences of T is replaced with U
////////////////////////////////////////////////////////////////////////////////

        template <class TList, class T, class U> struct ReplaceAll;
        
        template <class T, class U>
        struct ReplaceAll<NullType, T, U>
        {
            typedef NullType Result;
        };
        
        template <class T, class Tail, class U>
        struct ReplaceAll<Typelist<T, Tail>, T, U>
        {
            typedef Typelist<U, typename ReplaceAll<Tail, T, U>::Result> Result;
        };
        
        template <class Head, class Tail, class T, class U>
        struct ReplaceAll<Typelist<Head, Tail>, T, U>
        {
            typedef Typelist<Head,
                    typename ReplaceAll<Tail, T, U>::Result>
                Result;
        };

////////////////////////////////////////////////////////////////////////////////
// class template Reverse
// Reverses a typelist
// Invocation (TList is a typelist):
// Reverse<TList>::Result
// returns a typelist that is TList reversed
////////////////////////////////////////////////////////////////////////////////

        template <class TList> struct Reverse;
        
        template <>
        struct Reverse<NullType>
        {
            typedef NullType Result;
        };
        
        template <class Head, class Tail>
        struct Reverse< Typelist<Head, Tail> >
        {
            typedef typename Append<
                typename Reverse<Tail>::Result, Head>::Result Result;
        };

////////////////////////////////////////////////////////////////////////////////
// class template MostDerived
// Finds the type in a typelist that is the most derived from a given type
// Invocation (TList is a typelist, T is a type):
// MostDerived<TList, T>::Result
// returns the type in TList that's the most derived from T
////////////////////////////////////////////////////////////////////////////////

        template <class TList, class T> struct MostDerived;
        
        template <class T>
        struct MostDerived<NullType, T>
        {
            typedef T Result;
        };
        
        template <class Head, class Tail, class T>
        struct MostDerived<Typelist<Head, Tail>, T>
        {
        private:
            typedef typename MostDerived<Tail, T>::Result Candidate;
        public:
            typedef typename Select<
                SuperSubclass<Candidate,Head>::value,
                    Head, Candidate>::Result Result;
        };

////////////////////////////////////////////////////////////////////////////////
// class template DerivedToFront
// Arranges the types in a typelist so that the most derived types appear first
// Invocation (TList is a typelist):
// DerivedToFront<TList>::Result
// returns the reordered TList 
////////////////////////////////////////////////////////////////////////////////

        template <class TList> struct DerivedToFront;
        
        template <>
        struct DerivedToFront<NullType>
        {
            typedef NullType Result;
        };
        
        template <class Head, class Tail>
        struct DerivedToFront< Typelist<Head, Tail> >
        {
        private:
            typedef typename MostDerived<Tail, Head>::Result
                TheMostDerived;
            typedef typename Replace<Tail,
                TheMostDerived, Head>::Result Temp;
            typedef typename DerivedToFront<Temp>::Result L;
        public:
            typedef Typelist<TheMostDerived, L> Result;
        };
        
    }   // namespace TL
}   // namespace Loki


#endif // end file guardian

