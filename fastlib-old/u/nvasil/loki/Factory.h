////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2001 by Andrei Alexandrescu
// Copyright (c) 2005 by Peter Kuemmel
// This code DOES NOT accompany the book:
// Alexandrescu, Andrei. "Modern C++ Design: Generic Programming and Design
//     Patterns Applied". Copyright (c) 2001. Addison-Wesley.
//
// Code covered by the MIT License
// The authors make no representations about the suitability of this software
// for any purpose. It is provided "as is" without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////
#ifndef LOKI_FACTORYPARM_INC_
#define LOKI_FACTORYPARM_INC_

// $Id: Factory.h 788 2006-11-24 22:30:54Z clitte_bbt $


#include "LokiTypeInfo.h"
#include "Functor.h"
#include "AssocVector.h"
#include "SmallObj.h"
#include "Sequence.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4702)
//unreachable code if OnUnknownType throws an exception
#endif

/**
 * \defgroup	FactoriesGroup Factories
 * \defgroup	FactoryGroup Factory
 * \ingroup		FactoriesGroup
 * \brief		Implements a generic object factory.
 * 
 * <i>The Factory Method pattern is an object-oriented design pattern.
 * Like other creational patterns, it deals with the problem of creating objects
 * (products) without specifying the exact class of object that will be created.
 * Factory Method, one of the patterns from the Design Patterns book, handles
 * this problem by defining a separate method for creating the objects, which
 * subclasses can then override to specify the derived type of product that will
 * be created.
 * <br>
 * More generally, the term Factory Method is often used to refer to any method
 * whose main purpose is creation of objects.</i>
 * <div ALIGN="RIGHT"><a href="http://en.wikipedia.org/wiki/Factory_method_pattern">
 * Wikipedia</a></div>
 * 
 * Loki proposes a generic version of the Factory. Here is a typical use.<br>
 * <code><br>
 * 1. Factory< AbstractProduct, int > aFactory;<br>
 * 2. aFactory.Register( 1, createProductNull );<br>
 * 3. aFactory.CreateObject( 1 ); <br>
 * </code><br>
 * <br>
 * - 1. The declaration<br>
 * You want a Factory that produces AbstractProduct.<br>
 * The client will refer to a creation method through an int.<br>
 * - 2.The registration<br>
 * The code that will contribute to the Factory will now need to declare its
 * ProductCreator by registering them into the Factory.<br>
 * A ProductCreator is a just a function that will return the right object. ie <br>
 * <code>
 * Product* createProductNull()<br>             
 * {<br>
 *     return new Product<br>
 * }<br>
 * </code><br>
 * - 3. The use<br>
 * Now the client can create object by calling the Factory's CreateObject method
 * with the right identifier. If the ProductCreator were to have arguments
 * (<i>ie :Product* createProductParm( int a, int b )</i>)
 */

namespace Loki
{

/**
 * \defgroup	FactoryErrorPoliciesGroup Factory Error Policies
 * \ingroup		FactoryGroup
 * \brief		Manages the "Unknown Type" error in an object factory
 * 
 * \class DefaultFactoryError
 * \ingroup		FactoryErrorPoliciesGroup
 * \brief		Default policy that throws an exception		
 * 
 */

    template <typename IdentifierType, class AbstractProduct>
    struct DefaultFactoryError
    {
        struct Exception : public std::exception
        {
            const char* what() const throw() { return "Unknown Type"; }
        };

        static AbstractProduct* OnUnknownType(IdentifierType)
        {
            throw Exception();
        }
    };


#define LOKI_ENABLE_NEW_FACTORY_CODE
#ifdef LOKI_ENABLE_NEW_FACTORY_CODE


////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
////////////////////////////////////////////////////////////////////////////////

    struct FactoryImplBase
    {
        typedef EmptyType Parm1;
        typedef EmptyType Parm2;
        typedef EmptyType Parm3;
        typedef EmptyType Parm4;
        typedef EmptyType Parm5;
        typedef EmptyType Parm6;
        typedef EmptyType Parm7;
        typedef EmptyType Parm8;
        typedef EmptyType Parm9;
        typedef EmptyType Parm10;
        typedef EmptyType Parm11;
        typedef EmptyType Parm12;
        typedef EmptyType Parm13;
        typedef EmptyType Parm14;
        typedef EmptyType Parm15;
    };

    template <typename AP, typename Id, typename TList >
    struct FactoryImpl;

    template<typename AP, typename Id>
    struct FactoryImpl<AP, Id, NullType>
                : public FactoryImplBase
    {
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id & id ) = 0;
    };
template <typename AP, typename Id, typename P1 >
    struct FactoryImpl<AP,Id, Seq<P1> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1 ) = 0;
    };

    template<typename AP, typename Id, typename P1,typename P2 >
    struct FactoryImpl<AP, Id, Seq<P1, P2> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2 ) = 0;
    };

    template<typename AP, typename Id, typename P1,typename P2,typename P3 >
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3 ) = 0;
    };

    template<typename AP, typename Id, typename P1,typename P2,typename P3,typename P4 >
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4 ) = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5 >
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4, P5> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5 ) = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6>
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4, P5, P6> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6 )
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7>
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4, P5, P6, P7> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7 )
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8>
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4, P5, P6, P7, P8> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9>
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10>
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9,Parm10)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10,
    typename P11>
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        typedef typename TypeTraits<P11>::ParameterType Parm11;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9,Parm10,
                                Parm11)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10,
    typename P11,typename P12>
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        typedef typename TypeTraits<P11>::ParameterType Parm11;
        typedef typename TypeTraits<P12>::ParameterType Parm12;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9,Parm10,
                                Parm11,Parm12)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10,
    typename P11,typename P12,typename P13>
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        typedef typename TypeTraits<P11>::ParameterType Parm11;
        typedef typename TypeTraits<P12>::ParameterType Parm12;
        typedef typename TypeTraits<P13>::ParameterType Parm13;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9,Parm10,
                                Parm11,Parm12,Parm13)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10,
    typename P11,typename P12,typename P13,typename P14>
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        typedef typename TypeTraits<P11>::ParameterType Parm11;
        typedef typename TypeTraits<P12>::ParameterType Parm12;
        typedef typename TypeTraits<P13>::ParameterType Parm13;
        typedef typename TypeTraits<P14>::ParameterType Parm14;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm8,Parm10,
                                Parm11,Parm12,Parm13,Parm14)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10,
    typename P11,typename P12,typename P13,typename P14,typename P15 >
    struct FactoryImpl<AP, Id, Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15> >
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        typedef typename TypeTraits<P11>::ParameterType Parm11;
        typedef typename TypeTraits<P12>::ParameterType Parm12;
        typedef typename TypeTraits<P13>::ParameterType Parm13;
        typedef typename TypeTraits<P14>::ParameterType Parm14;
        typedef typename TypeTraits<P15>::ParameterType Parm15;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9,Parm10,
                                Parm11,Parm12,Parm13,Parm14,Parm15 )
        = 0;
    };

#ifndef LOKI_DISABLE_TYPELIST_MACROS

    template <typename AP, typename Id, typename P1 >
    struct FactoryImpl<AP,Id, LOKI_TYPELIST_1( P1 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1 ) = 0;
    };

    template<typename AP, typename Id, typename P1,typename P2 >
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_2( P1, P2 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2 ) = 0;
    };

    template<typename AP, typename Id, typename P1,typename P2,typename P3 >
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_3( P1, P2, P3 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3 ) = 0;
    };

    template<typename AP, typename Id, typename P1,typename P2,typename P3,typename P4 >
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_4( P1, P2, P3, P4 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4 ) = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5 >
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_5( P1, P2, P3, P4, P5 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5 ) = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6>
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_6( P1, P2, P3, P4, P5, P6 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6 )
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7>
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_7( P1, P2, P3, P4, P5, P6, P7 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7 )
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8>
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_8( P1, P2, P3, P4, P5, P6, P7, P8 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9>
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_9( P1, P2, P3, P4, P5, P6, P7, P8, P9 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10>
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_10( P1, P2, P3, P4, P5, P6, P7, P8, P9, P10 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9,Parm10)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10,
    typename P11>
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_11( P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        typedef typename TypeTraits<P11>::ParameterType Parm11;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9,Parm10,
                                Parm11)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10,
    typename P11,typename P12>
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_12( P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        typedef typename TypeTraits<P11>::ParameterType Parm11;
        typedef typename TypeTraits<P12>::ParameterType Parm12;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9,Parm10,
                                Parm11,Parm12)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10,
    typename P11,typename P12,typename P13>
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_13( P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        typedef typename TypeTraits<P11>::ParameterType Parm11;
        typedef typename TypeTraits<P12>::ParameterType Parm12;
        typedef typename TypeTraits<P13>::ParameterType Parm13;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9,Parm10,
                                Parm11,Parm12,Parm13)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10,
    typename P11,typename P12,typename P13,typename P14>
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_14( P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        typedef typename TypeTraits<P11>::ParameterType Parm11;
        typedef typename TypeTraits<P12>::ParameterType Parm12;
        typedef typename TypeTraits<P13>::ParameterType Parm13;
        typedef typename TypeTraits<P14>::ParameterType Parm14;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm8,Parm10,
                                Parm11,Parm12,Parm13,Parm14)
        = 0;
    };

    template<typename AP, typename Id,
    typename P1,typename P2,typename P3,typename P4,typename P5,
    typename P6,typename P7,typename P8,typename P9,typename P10,
    typename P11,typename P12,typename P13,typename P14,typename P15 >
    struct FactoryImpl<AP, Id, LOKI_TYPELIST_15( P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15 )>
                : public FactoryImplBase
    {
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        typedef typename TypeTraits<P10>::ParameterType Parm10;
        typedef typename TypeTraits<P11>::ParameterType Parm11;
        typedef typename TypeTraits<P12>::ParameterType Parm12;
        typedef typename TypeTraits<P13>::ParameterType Parm13;
        typedef typename TypeTraits<P14>::ParameterType Parm14;
        typedef typename TypeTraits<P15>::ParameterType Parm15;
        virtual ~FactoryImpl() {}
        virtual AP* CreateObject(const Id& id,Parm1, Parm2, Parm3, Parm4, Parm5,
                                Parm6, Parm7, Parm8, Parm9,Parm10,
                                Parm11,Parm12,Parm13,Parm14,Parm15 )
        = 0;
    };

#endif //LOKI_DISABLE_TYPELIST_MACROS


////////////////////////////////////////////////////////////////////////////////
///  \class Factory
///
///  \ingroup FactoryGroup
///  Implements a generic object factory.
///
///  Create functions can have up to 15 parameters.
///
///  \par Singleton lifetime when used with Loki::SingletonHolder
///  Because Factory uses internally Functors which inherits from
///  SmallObject you must use the singleton lifetime
///  \code Loki::LongevityLifetime::DieAsSmallObjectChild \endcode
///  Alternatively you could suppress for Functor the inheritance
///  from SmallObject by defining the macro:
/// \code LOKI_FUNCTOR_IS_NOT_A_SMALLOBJECT \endcode
////////////////////////////////////////////////////////////////////////////////
    template
    <
        class AbstractProduct,
        typename IdentifierType,
        typename CreatorParmTList = NullType,
        template<typename, class> class FactoryErrorPolicy = DefaultFactoryError
    >
    class Factory : public FactoryErrorPolicy<IdentifierType, AbstractProduct>
    {
        typedef FactoryImpl< AbstractProduct, IdentifierType, CreatorParmTList > Impl;

        typedef typename Impl::Parm1 Parm1;
        typedef typename Impl::Parm2 Parm2;
        typedef typename Impl::Parm3 Parm3;
        typedef typename Impl::Parm4 Parm4;
        typedef typename Impl::Parm5 Parm5;
        typedef typename Impl::Parm6 Parm6;
        typedef typename Impl::Parm7 Parm7;
        typedef typename Impl::Parm8 Parm8;
        typedef typename Impl::Parm9 Parm9;
        typedef typename Impl::Parm10 Parm10;
        typedef typename Impl::Parm11 Parm11;
        typedef typename Impl::Parm12 Parm12;
        typedef typename Impl::Parm13 Parm13;
        typedef typename Impl::Parm14 Parm14;
        typedef typename Impl::Parm15 Parm15;

        typedef Functor<AbstractProduct*, CreatorParmTList> ProductCreator;

        typedef AssocVector<IdentifierType, ProductCreator> IdToProductMap;

        IdToProductMap associations_;

    public:

        Factory()
            : associations_()
        {
        }

        ~Factory()
        {
            associations_.erase(associations_.begin(), associations_.end());
        }

        bool Register(const IdentifierType& id, ProductCreator creator)
        {
            return associations_.insert(
                         typename IdToProductMap::value_type(id, creator)).second != 0;
        }

        template <class PtrObj, typename CreaFn>
        bool Register(const IdentifierType& id, const PtrObj& p, CreaFn fn)
        {
            ProductCreator creator( p, fn );
            return associations_.insert(
                typename IdToProductMap::value_type(id, creator)).second != 0;
        }

        bool Unregister(const IdentifierType& id)
        {
            return associations_.erase(id) != 0;
        }

        std::vector<IdentifierType> RegisteredIds()
        {
            std::vector<IdentifierType> ids;
            for(typename IdToProductMap::iterator it = associations_.begin();
                it != associations_.end(); ++it)
            {
                ids.push_back(it->first);
            }
            return ids;
        }

        AbstractProduct* CreateObject(const IdentifierType& id)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1 p1)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1 p1, Parm2 p2)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1 p1, Parm2 p2, Parm3 p3)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4,p5 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
                                            Parm6 p6)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4,p5,p6 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
                                            Parm6 p6, Parm7 p7 )
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4,p5,p6,p7 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
                                            Parm6 p6, Parm7 p7, Parm8 p8)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4,p5,p6,p7,p8 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
                                            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4,p5,p6,p7,p8,p9 );
            return this->OnUnknownType(id);
        }
        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
                                            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9,Parm10 p10)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4,p5,p6,p7,p8,p9,p10 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1  p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5  p5,
                                            Parm6  p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10,
                                            Parm11 p11)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1  p1,  Parm2  p2, Parm3 p3, Parm4 p4, Parm5  p5,
                                            Parm6  p6,  Parm7  p7, Parm8 p8, Parm9 p9, Parm10 p10,
                                            Parm11 p11, Parm12 p12)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1  p1,  Parm2  p2,  Parm3  p3, Parm4 p4, Parm5  p5,
                                            Parm6  p6,  Parm7  p7,  Parm8  p8, Parm9 p9, Parm10 p10,
                                            Parm11 p11, Parm12 p12, Parm13 p13)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1  p1,  Parm2  p2,  Parm3  p3,  Parm4  p4, Parm5  p5,
                                            Parm6  p6,  Parm7  p7,  Parm8  p8,  Parm9  p9, Parm10 p10,
                                            Parm11 p11, Parm12 p12, Parm13 p13, Parm14 p14)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14 );
            return this->OnUnknownType(id);
        }

        AbstractProduct* CreateObject(const IdentifierType& id,
                                            Parm1  p1,  Parm2  p2,  Parm3  p3,  Parm4  p4,  Parm5  p5,
                                            Parm6  p6,  Parm7  p7,  Parm8  p8,  Parm9  p9,  Parm10 p10,
                                            Parm11 p11, Parm12 p12, Parm13 p13, Parm14 p14, Parm15 p15)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
                return (i->second)( p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15 );
            return this->OnUnknownType(id);
        }

    };

#else

    template
    <
        class AbstractProduct,
        typename IdentifierType,
        typename ProductCreator = AbstractProduct* (*)(),
        template<typename, class>
            class FactoryErrorPolicy = DefaultFactoryError
    >
    class Factory
        : public FactoryErrorPolicy<IdentifierType, AbstractProduct>
    {
    public:
        bool Register(const IdentifierType& id, ProductCreator creator)
        {
            return associations_.insert(
                typename IdToProductMap::value_type(id, creator)).second != 0;
        }

        bool Unregister(const IdentifierType& id)
        {
            return associations_.erase(id) != 0;
        }

        AbstractProduct* CreateObject(const IdentifierType& id)
        {
            typename IdToProductMap::iterator i = associations_.find(id);
            if (i != associations_.end())
            {
                return (i->second)();
            }
            return this->OnUnknownType(id);
        }

    private:
        typedef AssocVector<IdentifierType, ProductCreator> IdToProductMap;
        IdToProductMap associations_;
    };

#endif //#define ENABLE_NEW_FACTORY_CODE

/**
 *   \defgroup	CloneFactoryGroup Clone Factory
 *   \ingroup	FactoriesGroup
 *   \brief		Creates a copy from a polymorphic object.
 *
 *   \class		CloneFactory
 *   \ingroup	CloneFactoryGroup
 *   \brief		Creates a copy from a polymorphic object.
 */

    template
    <
        class AbstractProduct,
        class ProductCreator =
            AbstractProduct* (*)(const AbstractProduct*),
        template<typename, class>
            class FactoryErrorPolicy = DefaultFactoryError
    >
    class CloneFactory
        : public FactoryErrorPolicy<TypeInfo, AbstractProduct>
    {
    public:
        bool Register(const TypeInfo& ti, ProductCreator creator)
        {
            return associations_.insert(
                typename IdToProductMap::value_type(ti, creator)).second != 0;
        }

        bool Unregister(const TypeInfo& id)
        {
            return associations_.erase(id) != 0;
        }

        AbstractProduct* CreateObject(const AbstractProduct* model)
        {
            if (model == NULL)
            {
            	return NULL;
            }

            typename IdToProductMap::iterator i = 
            	associations_.find(typeid(*model));
            	
            if (i != associations_.end())
            {
                return (i->second)(model);
            }
            return this->OnUnknownType(typeid(*model));
        }

    private:
        typedef AssocVector<TypeInfo, ProductCreator> IdToProductMap;
        IdToProductMap associations_;
    };
        
} // namespace Loki


#ifdef _MSC_VER
#pragma warning( pop )
#endif

#endif // end file guardian

