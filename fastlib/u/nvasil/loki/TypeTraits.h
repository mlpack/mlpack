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
// The author or Addison-Wesley Longman make no representations about the 
//     suitability of this software for any purpose. It is provided "as is" 
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////
#ifndef LOKI_TYPETRAITS_INC_
#define LOKI_TYPETRAITS_INC_

// $Id: TypeTraits.h 749 2006-10-17 19:49:26Z syntheticpp $


#include "Typelist.h"
#include "Sequence.h"

#if (defined _MSC_VER) && (_MSC_VER < 1400)
#include <string>
#endif


#ifdef _MSC_VER
#pragma warning( push ) 
#pragma warning( disable : 4180 ) //qualifier applied to function type has no meaning; ignored
#endif

namespace Loki
{
////////////////////////////////////////////////////////////////////////////////
// class template IsCustomUnsignedInt
// Offers a means to integrate nonstandard built-in unsigned integral types
// (such as unsigned __int64 or unsigned long long int) with the TypeTraits 
//     class template defined below.
// Invocation: IsCustomUnsignedInt<T> where T is any type
// Defines 'value', an enum that is 1 iff T is a custom built-in unsigned
//     integral type
// Specialize this class template for nonstandard unsigned integral types
//     and define value = 1 in those specializations
////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct IsCustomUnsignedInt
    {
        enum { value = 0 };
    };        

////////////////////////////////////////////////////////////////////////////////
// class template IsCustomSignedInt
// Offers a means to integrate nonstandard built-in unsigned integral types
// (such as unsigned __int64 or unsigned long long int) with the TypeTraits 
//     class template defined below.
// Invocation: IsCustomSignedInt<T> where T is any type
// Defines 'value', an enum that is 1 iff T is a custom built-in signed
//     integral type
// Specialize this class template for nonstandard unsigned integral types
//     and define value = 1 in those specializations
////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct IsCustomSignedInt
    {
        enum { value = 0 };
    };        

////////////////////////////////////////////////////////////////////////////////
// class template IsCustomFloat
// Offers a means to integrate nonstandard floating point types with the
//     TypeTraits class template defined below.
// Invocation: IsCustomFloat<T> where T is any type
// Defines 'value', an enum that is 1 iff T is a custom built-in
//     floating point type
// Specialize this class template for nonstandard unsigned integral types
//     and define value = 1 in those specializations
////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct IsCustomFloat
    {
        enum { value = 0 };
    };        

////////////////////////////////////////////////////////////////////////////////
// Helper types for class template TypeTraits defined below
////////////////////////////////////////////////////////////////////////////////

    namespace Private
    {
#ifndef LOKI_DISABLE_TYPELIST_MACROS    
        typedef LOKI_TYPELIST_4(unsigned char, unsigned short int,unsigned int, unsigned long int) 
            StdUnsignedInts;
        typedef LOKI_TYPELIST_4(signed char, short int,int, long int) 
            StdSignedInts;
        typedef LOKI_TYPELIST_3(bool, char, wchar_t) 
            StdOtherInts;
        typedef LOKI_TYPELIST_3(float, double, long double) 
            StdFloats;
#else
        typedef Loki::Seq<unsigned char, unsigned short int,unsigned int, unsigned long int>::Type
            StdUnsignedInts;
        typedef Loki::Seq<signed char, short int,int, long int>::Type
            StdSignedInts;
        typedef Loki::Seq<bool, char, wchar_t>::Type
            StdOtherInts;
        typedef Loki::Seq<float, double, long double>::Type
            StdFloats;

#endif            
        template <typename U> struct AddPointer
        {
            typedef U* Result;
        };

        template <typename U> struct AddPointer<U&>
        {
            typedef U* Result;
        };

        template <class U> struct AddReference
        {
            typedef U & Result;
        };

        template <class U> struct AddReference<U &>
        {
            typedef U & Result;
        };

        template <> struct AddReference<void>
        {
            typedef NullType Result;
        };

        template <class U> struct AddParameterType
        {
            typedef const U & Result;
        };

        template <class U> struct AddParameterType<U &>
        {
            typedef U & Result;
        };

        template <> struct AddParameterType<void>
        {
            typedef NullType Result;
        };
    
        template <typename T>
        struct IsFunctionPointerRaw
        {enum{result = 0};};

        template <typename T>
        struct IsFunctionPointerRaw<T(*)()> 
        {enum {result = 1};};

        template <typename T, 
            typename P01>
        struct IsFunctionPointerRaw<T(*)(P01)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, P15)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19, typename P20>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, P20)> 
        {enum {result = 1};};

        template <typename T>
        struct IsFunctionPointerRaw<T(*)(
            ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01>
        struct IsFunctionPointerRaw<T(*)(
            P01, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, P15,
            ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, ...)> 
        {enum {result = 1};};

        template <typename T, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19, typename P20>
        struct IsFunctionPointerRaw<T(*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, P20,
            ...)> 
        {enum {result = 1};};
        
        
        template <typename T>
        struct IsMemberFunctionPointerRaw
        {enum{result = 0};};

        template <typename T, typename S>
        struct IsMemberFunctionPointerRaw<T (S::*)()> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01>
        struct IsMemberFunctionPointerRaw<T (S::*)(P01)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, P15)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19, typename P20>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, P20)> 
        {enum {result = 1};};

        template <typename T, typename S>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, P15,
            ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, ...)> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19, typename P20>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, P20,
            ...)> 
        {enum {result = 1};};

        // Const versions

        template <typename T, typename S>
        struct IsMemberFunctionPointerRaw<T (S::*)() const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01>
        struct IsMemberFunctionPointerRaw<T (S::*)(P01) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, P15) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19, typename P20>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, P20) const> 
        {enum {result = 1};};

        template <typename T, typename S>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, P15,
            ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, ...) const> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19, typename P20>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, P20,
            ...) const> 
        {enum {result = 1};};

        // Volatile versions

        template <typename T, typename S>
        struct IsMemberFunctionPointerRaw<T (S::*)() volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01>
        struct IsMemberFunctionPointerRaw<T (S::*)(P01) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, P15) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19, typename P20>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, P20) volatile> 
        {enum {result = 1};};

        template <typename T, typename S>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, P15,
            ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, ...) volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19, typename P20>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, P20,
            ...) volatile> 
        {enum {result = 1};};

        // Const volatile versions

        template <typename T, typename S>
        struct IsMemberFunctionPointerRaw<T (S::*)() const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01>
        struct IsMemberFunctionPointerRaw<T (S::*)(P01) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, P15) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19, typename P20>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, P20) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05,
            P06, P07, P08, P09, P10,
            P11, P12, P13, P14, P15,
            ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, ...) const volatile> 
        {enum {result = 1};};

        template <typename T, typename S, 
            typename P01, typename P02, typename P03, typename P04, typename P05,
            typename P06, typename P07, typename P08, typename P09, typename P10,
            typename P11, typename P12, typename P13, typename P14, typename P15,
            typename P16, typename P17, typename P18, typename P19, typename P20>
        struct IsMemberFunctionPointerRaw<T (S::*)(
            P01, P02, P03, P04, P05, 
            P06, P07, P08, P09, P10, 
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, P20,
            ...) const volatile> 
        {enum {result = 1};};
               
      }// namespace Private
        
////////////////////////////////////////////////////////////////////////////////
// class template TypeTraits
//
// Figures out at compile time various properties of any given type
// Invocations (T is a type, TypeTraits<T>::Propertie):
//
// - isPointer       : returns true if T is a pointer type
// - PointeeType     : returns the type to which T points if T is a pointer 
//                     type, NullType otherwise
// - isReference     : returns true if T is a reference type
// - ReferredType    : returns the type to which T refers if T is a reference 
//                     type, NullType otherwise
// - isMemberPointer : returns true if T is a pointer to member type
// - isStdUnsignedInt: returns true if T is a standard unsigned integral type
// - isStdSignedInt  : returns true if T is a standard signed integral type
// - isStdIntegral   : returns true if T is a standard integral type
// - isStdFloat      : returns true if T is a standard floating-point type
// - isStdArith      : returns true if T is a standard arithmetic type
// - isStdFundamental: returns true if T is a standard fundamental type
// - isUnsignedInt   : returns true if T is a unsigned integral type
// - isSignedInt     : returns true if T is a signed integral type
// - isIntegral      : returns true if T is a integral type
// - isFloat         : returns true if T is a floating-point type
// - isArith         : returns true if T is a arithmetic type
// - isFundamental   : returns true if T is a fundamental type
// - ParameterType   : returns the optimal type to be used as a parameter for 
//                     functions that take Ts
// - isConst         : returns true if T is a const-qualified type
// - NonConstType    : Type with removed 'const' qualifier from T, if any
// - isVolatile      : returns true if T is a volatile-qualified type
// - NonVolatileType : Type with removed 'volatile' qualifier from T, if any
// - UnqualifiedType : Type with removed 'const' and 'volatile' qualifiers from 
//                     T, if any
// - ConstParameterType: returns the optimal type to be used as a parameter 
//                       for functions that take 'const T's
//
////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    class TypeTraits
    {
    private:
    
        template <class U> struct ReferenceTraits
        {
            enum { result = false };
            typedef U ReferredType;
        };
        
        template <class U> struct ReferenceTraits<U&>
        {
            enum { result = true };
            typedef U ReferredType;
        };
               
        template <class U> struct PointerTraits
        {
            enum { result = false };
            typedef NullType PointeeType;
        };
        
        template <class U> struct PointerTraits<U*>
        {
            enum { result = true };
            typedef U PointeeType;
        };
        
        template <class U> struct PointerTraits<U*&>
        {
            enum { result = true };
            typedef U PointeeType;
        };
          
        template <class U> struct PToMTraits
        {
            enum { result = false };
        };
        
        template <class U, class V> struct PToMTraits<U V::*>
        {
            enum { result = true };
        };
        
        template <class U, class V> struct PToMTraits<U V::*&>
        {
            enum { result = true };
        };
        
        template <class U> struct FunctionPointerTraits
        {
            enum{ result = Private::IsFunctionPointerRaw<U>::result };
        };
        
        template <typename U> struct PToMFunctionTraits
        {
            enum{ result = Private::IsMemberFunctionPointerRaw<U>::result };
        };
         
        template <class U> struct UnConst
        {
            typedef U Result;
            enum { isConst = 0 };
        };
        
        template <class U> struct UnConst<const U>
        {
            typedef U Result;
            enum { isConst = 1 };
        };

        template <class U> struct UnConst<const U&>
        {
            typedef U& Result;
            enum { isConst = 1 };
        };
  
        template <class U> struct UnVolatile
        {
            typedef U Result;
            enum { isVolatile = 0 };
        };
       
        template <class U> struct UnVolatile<volatile U>
        {
            typedef U Result;
            enum { isVolatile = 1 };
        };

        template <class U> struct UnVolatile<volatile U&>
        {
            typedef U& Result;
            enum { isVolatile = 1 };
        };
        
    public:
        typedef typename UnConst<T>::Result 
            NonConstType;
        typedef typename UnVolatile<T>::Result 
            NonVolatileType;
        typedef typename UnVolatile<typename UnConst<T>::Result>::Result 
            UnqualifiedType;
        typedef typename PointerTraits<UnqualifiedType>::PointeeType 
            PointeeType;
        typedef typename ReferenceTraits<T>::ReferredType 
            ReferredType;

        enum { isConst          = UnConst<T>::isConst };
        enum { isVolatile       = UnVolatile<T>::isVolatile };
        enum { isReference      = ReferenceTraits<UnqualifiedType>::result };
        enum { isFunction       = FunctionPointerTraits<typename Private::AddPointer<T>::Result >::result };
        enum { isFunctionPointer= FunctionPointerTraits<
                                        typename ReferenceTraits<UnqualifiedType>::ReferredType >::result };
        enum { isMemberFunctionPointer= PToMFunctionTraits<
                                        typename ReferenceTraits<UnqualifiedType>::ReferredType >::result };
        enum { isMemberPointer  = PToMTraits<
                                        typename ReferenceTraits<UnqualifiedType>::ReferredType >::result ||
                                        isMemberFunctionPointer };
        enum { isPointer        = PointerTraits<
                                        typename ReferenceTraits<UnqualifiedType>::ReferredType >::result ||
                                        isFunctionPointer };
        
        enum { isStdUnsignedInt = TL::IndexOf<Private::StdUnsignedInts, UnqualifiedType>::value >= 0 ||
                                  TL::IndexOf<Private::StdUnsignedInts, 
                                        typename ReferenceTraits<UnqualifiedType>::ReferredType>::value >= 0};
        enum { isStdSignedInt   = TL::IndexOf<Private::StdSignedInts, UnqualifiedType>::value >= 0 ||
                                  TL::IndexOf<Private::StdSignedInts, 
                                        typename ReferenceTraits<UnqualifiedType>::ReferredType>::value >= 0};
        enum { isStdIntegral    = isStdUnsignedInt || isStdSignedInt ||
                                  TL::IndexOf<Private::StdOtherInts, UnqualifiedType>::value >= 0 ||
                                  TL::IndexOf<Private::StdOtherInts, 
                                        typename ReferenceTraits<UnqualifiedType>::ReferredType>::value >= 0};
        enum { isStdFloat       = TL::IndexOf<Private::StdFloats, UnqualifiedType>::value >= 0 ||
                                  TL::IndexOf<Private::StdFloats, 
                                        typename ReferenceTraits<UnqualifiedType>::ReferredType>::value >= 0};
        enum { isStdArith       = isStdIntegral || isStdFloat };
        enum { isStdFundamental = isStdArith || isStdFloat || Conversion<T, void>::sameType };
            
        enum { isUnsignedInt    = isStdUnsignedInt || IsCustomUnsignedInt<UnqualifiedType>::value };
        enum { isSignedInt      = isStdSignedInt || IsCustomSignedInt<UnqualifiedType>::value };
        enum { isIntegral       = isStdIntegral || isUnsignedInt || isSignedInt };
        enum { isFloat          = isStdFloat || IsCustomFloat<UnqualifiedType>::value };
        enum { isArith          = isIntegral || isFloat };
        enum { isFundamental    = isStdFundamental || isArith };
        
        typedef typename Select<isStdArith || isPointer || isMemberPointer, T, 
                typename Private::AddParameterType<T>::Result>::Result 
            ParameterType;
    };
}

#ifdef _MSC_VER
#pragma warning( pop )
#endif // _MSC_VER


#endif // end file guardian

