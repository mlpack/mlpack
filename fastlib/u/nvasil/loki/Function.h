////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2005 Peter Kümmel
// Permission to use, copy, modify, distribute and sell this software for any 
//     purpose is hereby granted without fee, provided that the above copyright 
//     notice appear in all copies and that both that copyright notice and this 
//     permission notice appear in supporting documentation.
// The author makes no representations about the 
//     suitability of this software for any purpose. It is provided "as is" 
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////
#ifndef LOKI_FUNCTION_INC_
#define LOKI_FUNCTION_INC_

// $Id: Function.h 750 2006-10-17 19:50:02Z syntheticpp $


#define LOKI_ENABLE_FUNCTION

#include <loki/Functor.h>
#include <loki/Sequence.h>

namespace Loki
{

    ////////////////////////////////////////////////////////////////////////////////
    ///  \struct Function
    ///
    ///  \ingroup FunctorGroup
    ///  Allows a boost/TR1 like usage of Functor.
    /// 
    ///  \par Usage
    ///
    ///      - free functions: e.g.  \code Function<int(int,int)> f(&freeFunction);
    ///                              \endcode
    ///      - member functions: e.g \code Function<int()> f(&object,&ObjectType::memberFunction); 
    ///                              \endcode
    ///
    ///  see also test/Function/FunctionTest.cpp (the modified test program from boost)
    ////////////////////////////////////////////////////////////////////////////////
    
    template<class R = void()>
    struct Function;


    template<class R>
    struct Function<R()> : public Functor<R>
    {
        typedef Functor<R> FBase;

        Function() : FBase() {}

        Function(const Function& func) : FBase() 
        {
            if( !func.empty()) 
                FBase::operator=(func);
        }
                
        // test on emptiness
        template<class R2> 
        Function(Function<R2()> func) : FBase() 
        {
            if(!func.empty())
                FBase::operator=(func);
        }
        
        // clear  by '= 0'
        Function(const int i) : FBase()
        { 
            if(i==0)
                FBase::clear();
            else
                throw std::runtime_error("Loki::Function(const int i): i!=0");
        }
        
        template<class Func>
        Function(Func func) : FBase(func) {}

        template<class Host, class Func>
        Function(const Host& host, const Func& func) : FBase(host,func) {}

    };


////////////////////////////////////////////////////////////////////////////////
// macros for the repetitions
////////////////////////////////////////////////////////////////////////////////

#define LOKI_FUNCTION_BODY                          \
                                                    \
        Function() : FBase() {}                     \
                                                    \
        Function(const Function& func) : FBase()    \
        {                                           \
            if( !func.empty())                      \
                FBase::operator=(func);             \
        }                                           \
                                                    \
        Function(const int i) : FBase()             \
        {                                           \
            if(i==0)                                \
                FBase::clear();                     \
            else                                    \
                throw std::runtime_error(           \
            "Loki::Function(const int i): i!=0");   \
        }                                           \
                                                    \
        template<class Func>                        \
        Function(Func func) : FBase(func) {}        \
                                                    \
        template<class Host, class Func>            \
        Function(const Host& host, const Func& func): FBase(host,func) {}

        
#define LOKI_FUNCTION_R2_CTOR_BODY          \
                                            \
        : FBase()                           \
        {                                   \
            if(!func.empty())               \
                FBase::operator=(func);     \
        }


////////////////////////////////////////////////////////////////////////////////
// repetitions
////////////////////////////////////////////////////////////////////////////////

    template<>
    struct Function<>
        : public Loki::Functor<>
    {
        typedef Functor<> FBase;
        
        template<class R2>
        Function(Function<R2()> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY // if compilation breaks here then 
                           // Function.h was not included before
                           // Functor.h, check your include order
                           // or define LOKI_ENABLE_FUNCTION 
    };

    template<class R,class P01>
    struct Function<R(P01)> 
        : public Loki::Functor<R, Seq<P01> >
    {
        typedef Functor<R, Seq<P01> > FBase;
        
        template<class R2,class Q01>
        Function(Function<R2(Q01)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,class P01,class P02>
    struct Function<R(P01,P02)> 
        : public Functor<R, Seq<P01,P02> >
    {
        typedef Functor<R, Seq<P01,P02> > FBase;

        template<class R2,class Q01, class Q02>
        Function(Function<R2(Q01,Q02)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,class P01,class P02, class P03>
    struct Function<R(P01,P02,P03)> 
        : public Functor<R, Seq<P01,P02,P03> >
    {
        typedef Functor<R, Seq<P01,P02,P03> > FBase;

        template<class R2,class Q01, class Q02,class Q03>
        Function(Function<R2(Q01,Q02,Q03)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,class P01,class P02, class P03,class P04>
    struct Function<R(P01,P02,P03,P04)> 
        : public Functor<R, Seq<P01,P02,P03,P04> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04> > FBase;

        template<class R2,class Q01,class Q02, class Q03,class Q04>
        Function(Function<R2(Q01,Q02,Q03,Q04)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,class P01,class P02, class P03,class P04,class P05>
    struct Function<R(P01,P02,P03,P04,P05)> 
        : public Functor<R, Seq<P01,P02,P03,P04,P05> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04,P05> > FBase;

        template<class R2,class Q01,class Q02, class Q03,class Q04,class Q05>
        Function(Function<R2(Q01,Q02,Q03,Q04,Q05)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,    class P01,class P02, class P03,class P04,class P05,
                        class P06>
    struct Function<R(P01,P02,P03,P04,P05,P06)> 
        : public Functor<R, Seq<P01,P02,P03,P04,P05,P06> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04,P05,P06> > FBase;
        
        template<class R2,    class Q01,class Q02, class Q03,class Q04,class Q05,
                            class Q06>
        Function(Function<R2(Q01,Q02,Q03,Q04,Q05,Q06)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,    class P01,class P02, class P03,class P04,class P05,
                        class P06,class P07>
    struct Function<R(P01,P02,P03,P04,P05,P06,P07)> 
        : public Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07> > FBase;

        template<class R2,    class Q01,class Q02, class Q03,class Q04,class Q05,
                            class Q06,class Q07>
        Function(Function<R2(Q01,Q02,Q03,Q04,Q05,Q06,Q07)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,    class P01,class P02, class P03,class P04,class P05,
                        class P06,class P07, class P08>
    struct Function<R(P01,P02,P03,P04,P05,P06,P07,P08)> 
        : public Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08> > FBase;
        
        template<class R2,    class Q01,class Q02, class Q03,class Q04,class Q05,
                            class Q06,class Q07, class Q08>
        Function(Function<R2(Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,    class P01,class P02, class P03,class P04,class P05,
                        class P06,class P07, class P08,class P09>
    struct Function<R(P01,P02,P03,P04,P05,P06,P07,P08,P09)> 
        : public Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09    > > FBase;
        
        template<class R2,    class Q01,class Q02, class Q03,class Q04,class Q05,
                            class Q06,class Q07, class Q08,class Q09>
        Function(Function<R2(Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08,Q09)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,    class P01,class P02, class P03,class P04,class P05,
                        class P06,class P07, class P08,class P09,class P10>
    struct Function<R(P01,P02,P03,P04,P05,P06,P07,P08,P09,P10)> 
        : public Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10> > FBase;
        
        template<class R2,    class Q01,class Q02, class Q03,class Q04,class Q05,
                            class Q06,class Q07, class Q08,class Q09,class Q10>
        Function(Function<R2(Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08,Q09,Q10)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,    class P01,class P02, class P03,class P04,class P05,
                        class P06,class P07, class P08,class P09,class P10,
                        class P11>
    struct Function<R(P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11)> 
            : public Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11> >FBase;
        
        template<class R2,    class Q01,class Q02, class Q03,class Q04,class Q05,
                            class Q06,class Q07, class Q08,class Q09,class Q10,
                            class Q11>
        Function(Function<R2(Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08,Q09,Q10,Q11)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,    class P01,class P02, class P03,class P04,class P05,
                        class P06,class P07, class P08,class P09,class P10,
                        class P11,class P12>
    struct Function<R(P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12)> 
        : public Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12> > FBase;
        
        template<class R2,    class Q01,class Q02, class Q03,class Q04,class Q05,
                            class Q06,class Q07, class Q08,class Q09,class Q10,
                            class Q11,class Q12>
        Function(Function<R2(Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08,Q09,Q10,Q11)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,    class P01,class P02, class P03,class P04,class P05,
                        class P06,class P07, class P08,class P09,class P10,
                        class P11,class P12, class P13>
    struct Function<R(P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12,P13)> 
        : public Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12,P13> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12,P13> > FBase;
        
        template<class R2,    class Q01,class Q02, class Q03,class Q04,class Q05,
                            class Q06,class Q07, class Q08,class Q09,class Q10,
                            class Q11,class Q12, class Q13>
        Function(Function<R2(Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08,Q09,Q10,Q11,Q12,Q13)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,    class P01,class P02, class P03,class P04,class P05,
                        class P06,class P07, class P08,class P09,class P10,
                        class P11,class P12, class P13,class P14>
    struct Function<R(P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12,P13,P14)> 
        : public Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12,P13,P14> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12,P13,P14> > FBase;
        template<class R2,    class Q01,class Q02, class Q03,class Q04,class Q05,
                            class Q06,class Q07, class Q08,class Q09,class Q10,
                            class Q11,class Q12, class Q13,class Q14>
        Function(Function<R2(Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08,Q09,Q10,Q11,Q12,Q13,Q14)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

    template<class R,    class P01,class P02, class P03,class P04,class P05,
                        class P06,class P07, class P08,class P09,class P10,
                        class P11,class P12, class P13,class P14,class P15>
    struct Function<R(P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12,P13,P14,P15)> 
        : public Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12,P13,P14,P15> >
    {
        typedef Functor<R, Seq<P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12,P13,P14,P15> > FBase;

        template<class R2,    class Q01,class Q02, class Q03,class Q04,class Q05,
                            class Q06,class Q07, class Q08,class Q09,class Q10,
                            class Q11,class Q12, class Q13,class Q14,class Q15>
        Function(Function<R2(Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08,Q09,Q10,Q11,Q12,Q13,Q14,Q15)> func) 
            LOKI_FUNCTION_R2_CTOR_BODY

        LOKI_FUNCTION_BODY
    };

}// namespace Loki

#endif // end file guardian

