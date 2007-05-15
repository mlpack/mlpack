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
#ifndef LOKI_FUNCTOR_INC_
#define LOKI_FUNCTOR_INC_

// $Id: Functor.h 750 2006-10-17 19:50:02Z syntheticpp $


#include "Typelist.h"
#include "Sequence.h"
#include "EmptyType.h"
#include "SmallObj.h"
#include "TypeTraits.h"
#include <typeinfo>
#include <memory>

///  \defgroup FunctorGroup Function objects

#ifndef LOKI_FUNCTOR_IS_NOT_A_SMALLOBJECT
//#define LOKI_FUNCTOR_IS_NOT_A_SMALLOBJECT
#endif

#ifndef LOKI_FUNCTORS_ARE_COMPARABLE
//#define LOKI_FUNCTORS_ARE_COMPARABLE
#endif


/// \namespace Loki
/// All classes of Loki are in the Loki namespace
namespace Loki
{
////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl (internal)
////////////////////////////////////////////////////////////////////////////////

    namespace Private
    {
        template <typename R, template <class, class> class ThreadingModel>
        struct FunctorImplBase 
#ifdef LOKI_FUNCTOR_IS_NOT_A_SMALLOBJECT
        {
#else
            : public SmallValueObject<ThreadingModel>
        {
            inline FunctorImplBase() :
                SmallValueObject<ThreadingModel>() {}
            inline FunctorImplBase(const FunctorImplBase&) :
                SmallValueObject<ThreadingModel>() {}
#endif

            typedef R ResultType;
            typedef FunctorImplBase<R, ThreadingModel> FunctorImplBaseType;

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


            virtual ~FunctorImplBase()
            {}

            virtual FunctorImplBase* DoClone() const = 0;

            template <class U>
            static U* Clone(U* pObj)
            {
                if (!pObj) return 0;
                U* pClone = static_cast<U*>(pObj->DoClone());
                assert(typeid(*pClone) == typeid(*pObj));
                return pClone;
            }


#ifdef LOKI_FUNCTORS_ARE_COMPARABLE

            virtual bool operator==(const FunctorImplBase&) const = 0;
           
#endif            
         
        };
    }
    
////////////////////////////////////////////////////////////////////////////////
// macro LOKI_DEFINE_CLONE_FUNCTORIMPL
// Implements the DoClone function for a functor implementation
////////////////////////////////////////////////////////////////////////////////

#define LOKI_DEFINE_CLONE_FUNCTORIMPL(Cls) \
    virtual Cls* DoClone() const { return new Cls(*this); }

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// The base class for a hierarchy of functors. The FunctorImpl class is not used
//     directly; rather, the Functor class manages and forwards to a pointer to
//     FunctorImpl
// You may want to derive your own functors from FunctorImpl.
// Specializations of FunctorImpl for up to 15 parameters follow
////////////////////////////////////////////////////////////////////////////////

    template <typename R, class TList, 
        template <class, class> class ThreadingModel = LOKI_DEFAULT_THREADING_NO_OBJ_LEVEL>
    class FunctorImpl;

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 0 (zero) parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, template <class, class> class ThreadingModel>
    class FunctorImpl<R, NullType, ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        virtual R operator()() = 0;
    };

    ////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 1 parameter
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, template <class, class> class ThreadingModel>
        class FunctorImpl<R, Seq<P1>, ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        virtual R operator()(Parm1) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 2 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, 
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, Seq<P1, P2>, ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        virtual R operator()(Parm1, Parm2) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 3 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, Seq<P1, P2, P3>, ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        virtual R operator()(Parm1, Parm2, Parm3) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 4 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, Seq<P1, P2, P3, P4>, ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 5 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, Seq<P1, P2, P3, P4, P5>, ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 6 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, Seq<P1, P2, P3, P4, P5, P6>, ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 7 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, Seq<P1, P2, P3, P4, P5, P6, P7>, ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 8 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, Seq<P1, P2, P3, P4, P5, P6, P7, P8>,
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 9 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9>,
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 10 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10>,
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 11 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10, typename P11,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R,
            Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11>,
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10, Parm11) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 12 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10, typename P11, typename P12,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R,
            Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12>,
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10, Parm11, Parm12) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 13 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10, typename P11, typename P12, typename P13,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R,
            Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13>,
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10, Parm11, Parm12, Parm13) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 14 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10, typename P11, typename P12, typename P13, typename P14,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R,
            Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13,
                P14>,
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10, Parm11, Parm12, Parm13, Parm14) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 15 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10, typename P11, typename P12, typename P13, typename P14,
        typename P15, template <class, class> class ThreadingModel>
    class FunctorImpl<R,
            Seq<P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13,
                P14, P15>,
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10, Parm11, Parm12, Parm13, Parm14,
            Parm15) = 0;
    };

#ifndef LOKI_DISABLE_TYPELIST_MACROS

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 1 parameter
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, template <class, class> class ThreadingModel>
    class FunctorImpl<R, LOKI_TYPELIST_1(P1), ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        virtual R operator()(Parm1) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 2 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, 
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, LOKI_TYPELIST_2(P1, P2), ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        virtual R operator()(Parm1, Parm2) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 3 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, LOKI_TYPELIST_3(P1, P2, P3), ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        virtual R operator()(Parm1, Parm2, Parm3) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 4 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, LOKI_TYPELIST_4(P1, P2, P3, P4), ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 5 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, LOKI_TYPELIST_5(P1, P2, P3, P4, P5), ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 6 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, LOKI_TYPELIST_6(P1, P2, P3, P4, P5, P6), ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 7 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, LOKI_TYPELIST_7(P1, P2, P3, P4, P5, P6, P7), ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 8 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, LOKI_TYPELIST_8(P1, P2, P3, P4, P5, P6, P7, P8),
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 9 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, LOKI_TYPELIST_9(P1, P2, P3, P4, P5, P6, P7, P8, P9),
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
        typedef typename TypeTraits<P1>::ParameterType Parm1;
        typedef typename TypeTraits<P2>::ParameterType Parm2;
        typedef typename TypeTraits<P3>::ParameterType Parm3;
        typedef typename TypeTraits<P4>::ParameterType Parm4;
        typedef typename TypeTraits<P5>::ParameterType Parm5;
        typedef typename TypeTraits<P6>::ParameterType Parm6;
        typedef typename TypeTraits<P7>::ParameterType Parm7;
        typedef typename TypeTraits<P8>::ParameterType Parm8;
        typedef typename TypeTraits<P9>::ParameterType Parm9;
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 10 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R, LOKI_TYPELIST_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10),
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 11 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10, typename P11,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R,
            LOKI_TYPELIST_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11),
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10, Parm11) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 12 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10, typename P11, typename P12,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R,
            LOKI_TYPELIST_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12),
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10, Parm11, Parm12) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 13 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10, typename P11, typename P12, typename P13,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R,
            LOKI_TYPELIST_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13),
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10, Parm11, Parm12, Parm13) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 14 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10, typename P11, typename P12, typename P13, typename P14,
        template <class, class> class ThreadingModel>
    class FunctorImpl<R,
            LOKI_TYPELIST_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13,
                P14),
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10, Parm11, Parm12, Parm13, Parm14) = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// class template FunctorImpl
// Specialization for 15 parameters
////////////////////////////////////////////////////////////////////////////////

    template <typename R, typename P1, typename P2, typename P3, typename P4,
        typename P5, typename P6, typename P7, typename P8, typename P9,
        typename P10, typename P11, typename P12, typename P13, typename P14,
        typename P15, template <class, class> class ThreadingModel>
    class FunctorImpl<R,
            LOKI_TYPELIST_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13,
                P14, P15),
            ThreadingModel>
        : public Private::FunctorImplBase<R, ThreadingModel>
    {
    public:
        typedef R ResultType;
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
        virtual R operator()(Parm1, Parm2, Parm3, Parm4, Parm5, Parm6, 
            Parm7, Parm8, Parm9, Parm10, Parm11, Parm12, Parm13, Parm14,
            Parm15) = 0;
    };

#endif //LOKI_DISABLE_TYPELIST_MACROS

////////////////////////////////////////////////////////////////////////////////
// class template FunctorHandler
// Wraps functors and pointers to functions
////////////////////////////////////////////////////////////////////////////////

    template <class ParentFunctor, typename Fun>
    class FunctorHandler
        : public ParentFunctor::Impl
    {
        typedef typename ParentFunctor::Impl Base;

    public:
        typedef typename Base::ResultType ResultType;
        typedef typename Base::Parm1 Parm1;
        typedef typename Base::Parm2 Parm2;
        typedef typename Base::Parm3 Parm3;
        typedef typename Base::Parm4 Parm4;
        typedef typename Base::Parm5 Parm5;
        typedef typename Base::Parm6 Parm6;
        typedef typename Base::Parm7 Parm7;
        typedef typename Base::Parm8 Parm8;
        typedef typename Base::Parm9 Parm9;
        typedef typename Base::Parm10 Parm10;
        typedef typename Base::Parm11 Parm11;
        typedef typename Base::Parm12 Parm12;
        typedef typename Base::Parm13 Parm13;
        typedef typename Base::Parm14 Parm14;
        typedef typename Base::Parm15 Parm15;
        
        FunctorHandler(const Fun& fun) : f_(fun) {}
        
        LOKI_DEFINE_CLONE_FUNCTORIMPL(FunctorHandler)


#ifdef LOKI_FUNCTORS_ARE_COMPARABLE


        bool operator==(const typename Base::FunctorImplBaseType& rhs) const
        {
            // there is no static information if Functor holds a member function 
            // or a free function; this is the main difference to tr1::function
            if(typeid(*this) != typeid(rhs))
                return false; // cannot be equal

            const FunctorHandler& fh = static_cast<const FunctorHandler&>(rhs);
            // if this line gives a compiler error, you are using a function object.
            // you need to implement bool MyFnObj::operator == (const MyFnObj&) const;
            return  f_==fh.f_;
        }
#endif
        // operator() implementations for up to 15 arguments
                
        ResultType operator()()
        { return f_(); }

        ResultType operator()(Parm1 p1)
        { return f_(p1); }
        
        ResultType operator()(Parm1 p1, Parm2 p2)
        { return f_(p1, p2); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3)
        { return f_(p1, p2, p3); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4)
        { return f_(p1, p2, p3, p4); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5)
        { return f_(p1, p2, p3, p4, p5); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6)
        { return f_(p1, p2, p3, p4, p5, p6); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7)
        { return f_(p1, p2, p3, p4, p5, p6, p7); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8)
        { return f_(p1, p2, p3, p4, p5, p6, p7, p8); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9)
        { return f_(p1, p2, p3, p4, p5, p6, p7, p8, p9); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10)
        { return f_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11)
        { return f_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12)
        { return f_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13)
        { return f_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13, Parm14 p14)
        {
            return f_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, 
                p14);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13, Parm14 p14, Parm15 p15)
        {
            return f_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, 
                p14, p15);
        }
        
    private:
        Fun f_;
    };
        
////////////////////////////////////////////////////////////////////////////////
// class template FunctorHandler
// Wraps pointers to member functions
////////////////////////////////////////////////////////////////////////////////

    template <class ParentFunctor, typename PointerToObj,
        typename PointerToMemFn>
    class MemFunHandler : public ParentFunctor::Impl
    {
        typedef typename ParentFunctor::Impl Base;

    public:
        typedef typename Base::ResultType ResultType;
        typedef typename Base::Parm1 Parm1;
        typedef typename Base::Parm2 Parm2;
        typedef typename Base::Parm3 Parm3;
        typedef typename Base::Parm4 Parm4;
        typedef typename Base::Parm5 Parm5;
        typedef typename Base::Parm6 Parm6;
        typedef typename Base::Parm7 Parm7;
        typedef typename Base::Parm8 Parm8;
        typedef typename Base::Parm9 Parm9;
        typedef typename Base::Parm10 Parm10;
        typedef typename Base::Parm11 Parm11;
        typedef typename Base::Parm12 Parm12;
        typedef typename Base::Parm13 Parm13;
        typedef typename Base::Parm14 Parm14;
        typedef typename Base::Parm15 Parm15;

        MemFunHandler(const PointerToObj& pObj, PointerToMemFn pMemFn) 
        : pObj_(pObj), pMemFn_(pMemFn)
        {}
        
        LOKI_DEFINE_CLONE_FUNCTORIMPL(MemFunHandler)


#ifdef LOKI_FUNCTORS_ARE_COMPARABLE

        bool operator==(const typename Base::FunctorImplBaseType& rhs) const
        {
            if(typeid(*this) != typeid(rhs))
                return false; // cannot be equal 

            const MemFunHandler& mfh = static_cast<const MemFunHandler&>(rhs);
            // if this line gives a compiler error, you are using a function object.
            // you need to implement bool MyFnObj::operator == (const MyFnObj&) const;
            return  pObj_==mfh.pObj_ && pMemFn_==mfh.pMemFn_;
        }
#endif   

        ResultType operator()()
        { return ((*pObj_).*pMemFn_)(); }

        ResultType operator()(Parm1 p1)
        { return ((*pObj_).*pMemFn_)(p1); }
        
        ResultType operator()(Parm1 p1, Parm2 p2)
        { return ((*pObj_).*pMemFn_)(p1, p2); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3)
        { return ((*pObj_).*pMemFn_)(p1, p2, p3); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4)
        { return ((*pObj_).*pMemFn_)(p1, p2, p3, p4); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5)
        { return ((*pObj_).*pMemFn_)(p1, p2, p3, p4, p5); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6)
        { return ((*pObj_).*pMemFn_)(p1, p2, p3, p4, p5, p6); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7)
        { return ((*pObj_).*pMemFn_)(p1, p2, p3, p4, p5, p6, p7); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8)
        { return ((*pObj_).*pMemFn_)(p1, p2, p3, p4, p5, p6, p7, p8); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9)
        { return ((*pObj_).*pMemFn_)(p1, p2, p3, p4, p5, p6, p7, p8, p9); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10)
        { return ((*pObj_).*pMemFn_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11)
        {
            return ((*pObj_).*pMemFn_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, 
                p11);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12)
        {
            return ((*pObj_).*pMemFn_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, 
                p11, p12);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13)
        {
            return ((*pObj_).*pMemFn_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, 
                p11, p12, p13);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13, Parm14 p14)
        {
            return ((*pObj_).*pMemFn_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, 
                p11, p12, p13, p14);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13, Parm14 p14, Parm15 p15)
        {
            return ((*pObj_).*pMemFn_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, 
                p11, p12, p13, p14, p15);
        }
        
    private:
        PointerToObj pObj_;
        PointerToMemFn pMemFn_;
    };
        
////////////////////////////////////////////////////////////////////////////////
// TR1 exception
//////////////////////////////////////////////////////////////////////////////////

#ifdef LOKI_ENABLE_FUNCTION

    class bad_function_call : public std::runtime_error
    {
    public:
        bad_function_call() : std::runtime_error("bad_function_call in Loki::Functor")
        {}
    };

#define LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL if(empty()) throw bad_function_call();

#else

#define LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL 

#endif

////////////////////////////////////////////////////////////////////////////////
///  \class Functor
///
///  \ingroup FunctorGroup
///  A generalized functor implementation with value semantics
///
/// \par Macro: LOKI_FUNCTOR_IS_NOT_A_SMALLOBJECT
/// Define 
/// \code LOKI_FUNCTOR_IS_NOT_A_SMALLOBJECT \endcode
/// to avoid static instantiation/delete 
/// order problems.
/// It often helps against crashes when using static Functors and multi threading.
/// Defining also removes problems when unloading Dlls which hosts
/// static Functor objects.
///
/// \par Macro: LOKI_FUNCTORS_ARE_COMPARABLE
/// To enable the operator== define the macro
/// \code LOKI_FUNCTORS_ARE_COMPARABLE \endcode
/// The macro is disabled by default, because it breaks compiling functor 
/// objects  which have no operator== implemented, keep in mind when you enable
/// operator==.
////////////////////////////////////////////////////////////////////////////////
    template <typename R = void, class TList = NullType,
        template<class, class> class ThreadingModel = LOKI_DEFAULT_THREADING_NO_OBJ_LEVEL>
    class Functor
    {
    public:
        // Handy type definitions for the body type
        typedef FunctorImpl<R, TList, ThreadingModel> Impl;
        typedef R ResultType;
        typedef TList ParmList;
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

        // Member functions

        Functor() : spImpl_(0)
        {}
        
        Functor(const Functor& rhs) : spImpl_(Impl::Clone(rhs.spImpl_.get()))
        {}
        
        Functor(std::auto_ptr<Impl> spImpl) : spImpl_(spImpl)
        {}
        
        template <typename Fun>
        Functor(Fun fun)
        : spImpl_(new FunctorHandler<Functor, Fun>(fun))
        {}

        template <class PtrObj, typename MemFn>
        Functor(const PtrObj& p, MemFn memFn)
        : spImpl_(new MemFunHandler<Functor, PtrObj, MemFn>(p, memFn))
        {}

        typedef Impl * (std::auto_ptr<Impl>::*unspecified_bool_type)() const;

        operator unspecified_bool_type() const
        {
            return spImpl_.get() ? &std::auto_ptr<Impl>::get : 0;
        }

        Functor& operator=(const Functor& rhs)
        {
            Functor copy(rhs);
            // swap auto_ptrs by hand
            Impl* p = spImpl_.release();
            spImpl_.reset(copy.spImpl_.release());
            copy.spImpl_.reset(p);
            return *this;
        }

#ifdef LOKI_ENABLE_FUNCTION

        bool empty() const
        {
            return spImpl_.get() == 0;
        }

        void clear()
        {
            spImpl_.reset(0);
        }
#endif

#ifdef LOKI_FUNCTORS_ARE_COMPARABLE

        bool operator==(const Functor& rhs) const
        {
            if(spImpl_.get()==0 && rhs.spImpl_.get()==0)
                return true;
            if(spImpl_.get()!=0 && rhs.spImpl_.get()!=0)
                return *spImpl_.get() == *rhs.spImpl_.get();
            else
                return false;
        }

        bool operator!=(const Functor& rhs) const
        {
            return !(*this==rhs);
        }
#endif

        // operator() implementations for up to 15 arguments

        ResultType operator()() const
        {
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(); 
        }

        ResultType operator()(Parm1 p1) const
        { 
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1); 
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2) const
        {    
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2); 
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3) const
        {    
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3); 
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4) const
        { 
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4); 
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5) const
        { 
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4, p5); 
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6) const
        { 
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4, p5, p6); 
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7) const
        { 
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4, p5, p6, p7); 
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8) const
        { 
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4, p5, p6, p7, p8); 
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9) const
        { 
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4, p5, p6, p7, p8, p9); 
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10) const
        { 
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10); 
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11) const
        { 
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11); 
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12) const
        {
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, 
                p12);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13) const
        {
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
            p12, p13);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13, Parm14 p14) const
        {
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, 
                p12, p13, p14);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13, Parm14 p14, Parm15 p15) const
        {
            LOKI_FUNCTION_THROW_BAD_FUNCTION_CALL
            return (*spImpl_)(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, 
                p12, p13, p14, p15);
        }

    private:
        std::auto_ptr<Impl> spImpl_;
    };
    

////////////////////////////////////////////////////////////////////////////////
//  
//  BindersFirst and Chainer 
//
////////////////////////////////////////////////////////////////////////////////

    namespace Private
    {
        template <class Fctor> struct BinderFirstTraits;

        template <typename R, class TList, template <class, class> class ThreadingModel>
        struct BinderFirstTraits< Functor<R, TList, ThreadingModel> >
        {
            typedef Functor<R, TList, ThreadingModel> OriginalFunctor;

            typedef typename TL::Erase<TList,typename TL::TypeAt<TList, 0>::Result>
                             ::Result
                    ParmList;

            typedef typename TL::TypeAt<TList, 0>::Result OriginalParm1;

            typedef Functor<R, ParmList, ThreadingModel> BoundFunctorType;

            typedef typename BoundFunctorType::Impl Impl;

        };  


        template<class T>
        struct BinderFirstBoundTypeStorage;

        template<class T>
        struct BinderFirstBoundTypeStorage
        {
            typedef typename TypeTraits<T>::ParameterType RefOrValue;
        };
        
        template <typename R, class TList, template <class, class> class ThreadingModel>
        struct BinderFirstBoundTypeStorage< Functor<R, TList, ThreadingModel> >
        {
            typedef Functor<R, TList, ThreadingModel> OriginalFunctor;
            typedef const typename TypeTraits<OriginalFunctor>::ReferredType RefOrValue;
        };  


    } // namespace Private

////////////////////////////////////////////////////////////////////////////////
///  \class BinderFirst
///  
///  \ingroup FunctorGroup
///  Binds the first parameter of a Functor object to a specific value
////////////////////////////////////////////////////////////////////////////////

    template <class OriginalFunctor>
    class BinderFirst 
        : public Private::BinderFirstTraits<OriginalFunctor>::Impl
    {
        typedef typename Private::BinderFirstTraits<OriginalFunctor>::Impl Base;
        typedef typename OriginalFunctor::ResultType ResultType;

        typedef typename OriginalFunctor::Parm1 BoundType;

        typedef typename Private::BinderFirstBoundTypeStorage<
                             typename Private::BinderFirstTraits<OriginalFunctor>
                             ::OriginalParm1>
                         ::RefOrValue
                BoundTypeStorage;
                        
        typedef typename OriginalFunctor::Parm2 Parm1;
        typedef typename OriginalFunctor::Parm3 Parm2;
        typedef typename OriginalFunctor::Parm4 Parm3;
        typedef typename OriginalFunctor::Parm5 Parm4;
        typedef typename OriginalFunctor::Parm6 Parm5;
        typedef typename OriginalFunctor::Parm7 Parm6;
        typedef typename OriginalFunctor::Parm8 Parm7;
        typedef typename OriginalFunctor::Parm9 Parm8;
        typedef typename OriginalFunctor::Parm10 Parm9;
        typedef typename OriginalFunctor::Parm11 Parm10;
        typedef typename OriginalFunctor::Parm12 Parm11;
        typedef typename OriginalFunctor::Parm13 Parm12;
        typedef typename OriginalFunctor::Parm14 Parm13;
        typedef typename OriginalFunctor::Parm15 Parm14;
        typedef EmptyType Parm15;

    public:
        
        BinderFirst(const OriginalFunctor& fun, BoundType bound)
        : f_(fun), b_(bound)
        {}

        LOKI_DEFINE_CLONE_FUNCTORIMPL(BinderFirst)

#ifdef LOKI_FUNCTORS_ARE_COMPARABLE
        
        bool operator==(const typename Base::FunctorImplBaseType& rhs) const
        {
            if(typeid(*this) != typeid(rhs))
                return false; // cannot be equal 
            // if this line gives a compiler error, you are using a function object.
            // you need to implement bool MyFnObj::operator == (const MyFnObj&) const;
            return    f_ == ((static_cast<const BinderFirst&> (rhs)).f_) &&
                      b_ == ((static_cast<const BinderFirst&> (rhs)).b_);
        }
#endif

        // operator() implementations for up to 15 arguments
                
        ResultType operator()()
        { return f_(b_); }

        ResultType operator()(Parm1 p1)
        { return f_(b_, p1); }
        
        ResultType operator()(Parm1 p1, Parm2 p2)
        { return f_(b_, p1, p2); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3)
        { return f_(b_, p1, p2, p3); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4)
        { return f_(b_, p1, p2, p3, p4); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5)
        { return f_(b_, p1, p2, p3, p4, p5); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6)
        { return f_(b_, p1, p2, p3, p4, p5, p6); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7)
        { return f_(b_, p1, p2, p3, p4, p5, p6, p7); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8)
        { return f_(b_, p1, p2, p3, p4, p5, p6, p7, p8); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9)
        { return f_(b_, p1, p2, p3, p4, p5, p6, p7, p8, p9); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10)
        { return f_(b_, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11)
        { return f_(b_, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12)
        { return f_(b_, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13)
        { return f_(b_, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13, Parm14 p14)
        {
            return f_(b_, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, 
                p14);
        }
        
    private:
        OriginalFunctor f_;
        BoundTypeStorage b_;
    };
    
////////////////////////////////////////////////////////////////////////////////
///  Binds the first parameter of a Functor object to a specific value
///  \ingroup FunctorGroup
////////////////////////////////////////////////////////////////////////////////

    template <class Fctor>
    typename Private::BinderFirstTraits<Fctor>::BoundFunctorType
    BindFirst(
        const Fctor& fun, 
        typename Fctor::Parm1 bound)
    {
        typedef typename Private::BinderFirstTraits<Fctor>::BoundFunctorType
            Outgoing;
        
        return Outgoing(std::auto_ptr<typename Outgoing::Impl>(
            new BinderFirst<Fctor>(fun, bound)));
    }

////////////////////////////////////////////////////////////////////////////////
///  \class Chainer
///
///  \ingroup FunctorGroup
///   Chains two functor calls one after another
////////////////////////////////////////////////////////////////////////////////

    template <typename Fun1, typename Fun2>
    class Chainer : public Fun2::Impl
    {
        typedef Fun2 Base;

    public:
        typedef typename Base::ResultType ResultType;
        typedef typename Base::Parm1 Parm1;
        typedef typename Base::Parm2 Parm2;
        typedef typename Base::Parm3 Parm3;
        typedef typename Base::Parm4 Parm4;
        typedef typename Base::Parm5 Parm5;
        typedef typename Base::Parm6 Parm6;
        typedef typename Base::Parm7 Parm7;
        typedef typename Base::Parm8 Parm8;
        typedef typename Base::Parm9 Parm9;
        typedef typename Base::Parm10 Parm10;
        typedef typename Base::Parm11 Parm11;
        typedef typename Base::Parm12 Parm12;
        typedef typename Base::Parm13 Parm13;
        typedef typename Base::Parm14 Parm14;
        typedef typename Base::Parm15 Parm15;
        
        Chainer(const Fun1& fun1, const Fun2& fun2) : f1_(fun1), f2_(fun2) {}

        LOKI_DEFINE_CLONE_FUNCTORIMPL(Chainer)

#ifdef LOKI_FUNCTORS_ARE_COMPARABLE
                
        bool operator==(const typename Base::Impl::FunctorImplBaseType& rhs) const
        {
            if(typeid(*this) != typeid(rhs))
                return false; // cannot be equal 
            // if this line gives a compiler error, you are using a function object.
            // you need to implement bool MyFnObj::operator == (const MyFnObj&) const;
            return    f1_ == ((static_cast<const Chainer&> (rhs)).f2_) &&
                      f2_ == ((static_cast<const Chainer&> (rhs)).f1_);
        }
#endif

        // operator() implementations for up to 15 arguments

        ResultType operator()()
        { return f1_(), f2_(); }

        ResultType operator()(Parm1 p1)
        { return f1_(p1), f2_(p1); }
        
        ResultType operator()(Parm1 p1, Parm2 p2)
        { return f1_(p1, p2), f2_(p1, p2); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3)
        { return f1_(p1, p2, p3), f2_(p1, p2, p3); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4)
        { return f1_(p1, p2, p3, p4), f2_(p1, p2, p3, p4); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5)
        { return f1_(p1, p2, p3, p4, p5), f2_(p1, p2, p3, p4, p5); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6)
        { return f1_(p1, p2, p3, p4, p5, p6), f2_(p1, p2, p3, p4, p5, p6); }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7)
        {
            return f1_(p1, p2, p3, p4, p5, p6, p7),
                f2_(p1, p2, p3, p4, p5, p6, p7);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8)
        {
            return f1_(p1, p2, p3, p4, p5, p6, p7, p8),
                f2_(p1, p2, p3, p4, p5, p6, p7, p8);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9)
        {
            return f1_(p1, p2, p3, p4, p5, p6, p7, p8, p9),
                f2_(p1, p2, p3, p4, p5, p6, p7, p8, p9);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10)
        {
            return f1_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10),
                f2_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11)
        {
            return f1_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11),
                f2_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12)
        {
            return f1_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12),
                f2_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13)
        {
            return f1_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13),
                f2_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13, Parm14 p14)
        {
            return f1_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, 
                    p14),
                f2_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, 
                   p14);
        }
        
        ResultType operator()(Parm1 p1, Parm2 p2, Parm3 p3, Parm4 p4, Parm5 p5,
            Parm6 p6, Parm7 p7, Parm8 p8, Parm9 p9, Parm10 p10, Parm11 p11,
            Parm12 p12, Parm13 p13, Parm14 p14, Parm15 p15)
        {
            return f1_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, 
                    p14, p15),
                f2_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, 
                    p14, p15);
        }
        
    private:
        Fun1 f1_;
        Fun2 f2_;
    };
    
////////////////////////////////////////////////////////////////////////////////
///  Chains two functor calls one after another
///  \ingroup FunctorGroup
////////////////////////////////////////////////////////////////////////////////


    template <class Fun1, class Fun2>
    Fun2 Chain(
        const Fun1& fun1,
        const Fun2& fun2)
    {
        return Fun2(std::auto_ptr<typename Fun2::Impl>(
            new Chainer<Fun1, Fun2>(fun1, fun2)));
    }

} // namespace Loki


#endif // end file guardian

