////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2000 Andrei Alexandrescu
// Copyright (c) 2000 Petru Marginean
// Copyright (c) 2005 Joshua Lehrer
//
// Permission to use, copy, modify, distribute and sell this software for any 
//     purpose is hereby granted without fee, provided that the above copyright 
//     notice appear in all copies and that both that copyright notice and this 
//     permission notice appear in supporting documentation.
// The author makes no representations about the 
//     suitability of this software for any purpose. It is provided "as is" 
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////
#ifndef LOKI_SCOPEGUARD_INC_
#define LOKI_SCOPEGUARD_INC_

// $Id: ScopeGuard.h 799 2006-12-20 00:37:13Z rich_sposato $


#include <loki/RefToValue.h>

/// \defgroup ExceptionGroup Exception-safe code

namespace Loki
{

    ////////////////////////////////////////////////////////////////
    ///
    /// \class ScopeGuardImplBase
    /// \ingroup ExceptionGroup
    ///
    /// Base class used by all ScopeGuard implementations.  All commonly used
    /// functions are in this class (e.g. - Dismiss and SafeExecute).
    ///
    /// See Andrei's and Petru Marginean's CUJ article
    /// http://www.cuj.com/documents/s=8000/cujcexp1812alexandr/alexandr.htm
    ///
    /// Changes to the original code by Joshua Lehrer:
    /// http://www.lehrerfamily.com/scopeguard.html
    ////////////////////////////////////////////////////////////////

    class ScopeGuardImplBase
    {
        /// Copy-assignment operator is not implemented and private.
        ScopeGuardImplBase& operator =(const ScopeGuardImplBase&);

    protected:

        ~ScopeGuardImplBase()
        {}

        /// Copy-constructor takes over responsibility from other ScopeGuard.
        ScopeGuardImplBase(const ScopeGuardImplBase& other) throw() 
            : dismissed_(other.dismissed_)
        {
            other.Dismiss();
        }

        template <typename J>
        static void SafeExecute(J& j) throw() 
        {
            if (!j.dismissed_)
                try
                {
                    j.Execute();
                }
                catch(...)
                {}
        }
        
        mutable bool dismissed_;

    public:
        ScopeGuardImplBase() throw() : dismissed_(false) 
        {}

        void Dismiss() const throw() 
        {
            dismissed_ = true;
        }
    };

    ////////////////////////////////////////////////////////////////
    ///
    /// \typedef typedef const ScopeGuardImplBase& ScopeGuard
    /// \ingroup ExceptionGroup
    ///
    ////////////////////////////////////////////////////////////////

    typedef const ScopeGuardImplBase& ScopeGuard;

    ////////////////////////////////////////////////////////////////
    ///
    /// \class ScopeGuardImpl0
    /// \ingroup ExceptionGroup
    ///
    /// Implementation class for a standalone function or class static function
    /// with no parameters.  ScopeGuard ignores any value returned from the
    /// call within the Execute function.
    ///
    /// This class has a single standalone helper function, MakeGuard which
    /// creates and returns a ScopeGuard.
    ///
    ////////////////////////////////////////////////////////////////

    template <typename F>
    class ScopeGuardImpl0 : public ScopeGuardImplBase
    {
    public:
        static ScopeGuardImpl0<F> MakeGuard(F fun)
        {
            return ScopeGuardImpl0<F>(fun);
        }

        ~ScopeGuardImpl0() throw() 
        {
            SafeExecute(*this);
        }

        void Execute() 
        {
            fun_();
        }

    protected:
        ScopeGuardImpl0(F fun) : fun_(fun) 
        {}

        F fun_;
    };

    template <typename F> 
    inline ScopeGuardImpl0<F> MakeGuard(F fun)
    {
        return ScopeGuardImpl0<F>::MakeGuard(fun);
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// \class ScopeGuardImpl1
    /// \ingroup ExceptionGroup
    ///
    /// Implementation class for a standalone function or class static function
    /// with one parameter.  Each parameter is copied by value - use
    /// ::Loki::ByRef if you must use a reference instead.  ScopeGuard ignores
    /// any value returned from the call within the Execute function.
    ///
    /// This class has a single standalone helper function, MakeGuard which
    /// creates and returns a ScopeGuard.
    ///
    ////////////////////////////////////////////////////////////////

    template <typename F, typename P1>
    class ScopeGuardImpl1 : public ScopeGuardImplBase
    {
    public:
        static ScopeGuardImpl1<F, P1> MakeGuard(F fun, P1 p1)
        {
            return ScopeGuardImpl1<F, P1>(fun, p1);
        }

        ~ScopeGuardImpl1() throw() 
        {
            SafeExecute(*this);
        }

        void Execute()
        {
            fun_(p1_);
        }

    protected:
        ScopeGuardImpl1(F fun, P1 p1) : fun_(fun), p1_(p1) 
        {}

        F fun_;
        const P1 p1_;
    };

    template <typename F, typename P1> 
    inline ScopeGuardImpl1<F, P1> MakeGuard(F fun, P1 p1)
    {
        return ScopeGuardImpl1<F, P1>::MakeGuard(fun, p1);
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// \class ScopeGuardImpl2
    /// \ingroup ExceptionGroup
    ///
    /// Implementation class for a standalone function or class static function
    /// with two parameters.  Each parameter is copied by value - use
    /// ::Loki::ByRef if you must use a reference instead.  ScopeGuard ignores
    /// any value returned from the call within the Execute function.
    ///
    /// This class has a single standalone helper function, MakeGuard which
    /// creates and returns a ScopeGuard.
    ///
    ////////////////////////////////////////////////////////////////

    template <typename F, typename P1, typename P2>
    class ScopeGuardImpl2: public ScopeGuardImplBase
    {
    public:
        static ScopeGuardImpl2<F, P1, P2> MakeGuard(F fun, P1 p1, P2 p2)
        {
            return ScopeGuardImpl2<F, P1, P2>(fun, p1, p2);
        }

        ~ScopeGuardImpl2() throw() 
        {
            SafeExecute(*this);
        }

        void Execute()
        {
            fun_(p1_, p2_);
        }

    protected:
        ScopeGuardImpl2(F fun, P1 p1, P2 p2) : fun_(fun), p1_(p1), p2_(p2) 
        {}

        F fun_;
        const P1 p1_;
        const P2 p2_;
    };

    template <typename F, typename P1, typename P2>
    inline ScopeGuardImpl2<F, P1, P2> MakeGuard(F fun, P1 p1, P2 p2)
    {
        return ScopeGuardImpl2<F, P1, P2>::MakeGuard(fun, p1, p2);
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// \class ScopeGuardImpl3
    /// \ingroup ExceptionGroup
    ///
    /// Implementation class for a standalone function or class static function
    /// with three parameters.  Each parameter is copied by value - use
    /// ::Loki::ByRef if you must use a reference instead.  ScopeGuard ignores
    /// any value returned from the call within the Execute function.
    ///
    /// This class has a single standalone helper function, MakeGuard which
    /// creates and returns a ScopeGuard.
    ///
    ////////////////////////////////////////////////////////////////

    template <typename F, typename P1, typename P2, typename P3>
    class ScopeGuardImpl3 : public ScopeGuardImplBase
    {
    public:
        static ScopeGuardImpl3<F, P1, P2, P3> MakeGuard(F fun, P1 p1, P2 p2, P3 p3)
        {
            return ScopeGuardImpl3<F, P1, P2, P3>(fun, p1, p2, p3);
        }

        ~ScopeGuardImpl3() throw() 
        {
            SafeExecute(*this);
        }

        void Execute()
        {
            fun_(p1_, p2_, p3_);
        }

    protected:
        ScopeGuardImpl3(F fun, P1 p1, P2 p2, P3 p3) : fun_(fun), p1_(p1), p2_(p2), p3_(p3) 
        {}

        F fun_;
        const P1 p1_;
        const P2 p2_;
        const P3 p3_;
    };

    template <typename F, typename P1, typename P2, typename P3>
    inline ScopeGuardImpl3<F, P1, P2, P3> MakeGuard(F fun, P1 p1, P2 p2, P3 p3)
    {
        return ScopeGuardImpl3<F, P1, P2, P3>::MakeGuard(fun, p1, p2, p3);
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// \class ScopeGuardImpl4
    /// \ingroup ExceptionGroup
    ///
    /// Implementation class for a standalone function or class static function
    /// with four parameters.  Each parameter is copied by value - use
    /// ::Loki::ByRef if you must use a reference instead.  ScopeGuard ignores
    /// any value returned from the call within the Execute function.
    ///
    /// This class has a single standalone helper function, MakeGuard which
    /// creates and returns a ScopeGuard.
    ///
    ////////////////////////////////////////////////////////////////

    template < typename F, typename P1, typename P2, typename P3, typename P4 >
    class ScopeGuardImpl4 : public ScopeGuardImplBase
    {
    public:
        static ScopeGuardImpl4< F, P1, P2, P3, P4 > MakeGuard(
            F fun, P1 p1, P2 p2, P3 p3, P4 p4 )
        {
            return ScopeGuardImpl4< F, P1, P2, P3, P4 >( fun, p1, p2, p3, p4 );
        }

        ~ScopeGuardImpl4() throw() 
        {
            SafeExecute( *this );
        }

        void Execute()
        {
            fun_( p1_, p2_, p3_, p4_ );
        }

    protected:
        ScopeGuardImpl4( F fun, P1 p1, P2 p2, P3 p3, P4 p4 ) :
             fun_( fun ), p1_( p1 ), p2_( p2 ), p3_( p3 ), p4_( p4 )
        {}

        F fun_;
        const P1 p1_;
        const P2 p2_;
        const P3 p3_;
        const P4 p4_;
    };

    template < typename F, typename P1, typename P2, typename P3, typename P4 >
    inline ScopeGuardImpl4< F, P1, P2, P3, P4 > MakeGuard( F fun, P1 p1, P2 p2, P3 p3, P4 p4 )
    {
        return ScopeGuardImpl4< F, P1, P2, P3, P4 >::MakeGuard( fun, p1, p2, p3, p4 );
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// \class ScopeGuardImpl5
    /// \ingroup ExceptionGroup
    ///
    /// Implementation class for a standalone function or class static function
    /// with five parameters.  Each parameter is copied by value - use
    /// ::Loki::ByRef if you must use a reference instead.  ScopeGuard ignores
    /// any value returned from the call within the Execute function.
    ///
    /// This class has a single standalone helper function, MakeGuard which
    /// creates and returns a ScopeGuard.
    ///
    ////////////////////////////////////////////////////////////////

    template < typename F, typename P1, typename P2, typename P3, typename P4, typename P5 >
    class ScopeGuardImpl5 : public ScopeGuardImplBase
    {
    public:
        static ScopeGuardImpl5< F, P1, P2, P3, P4, P5 > MakeGuard(
            F fun, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5 )
        {
            return ScopeGuardImpl5< F, P1, P2, P3, P4, P5 >( fun, p1, p2, p3, p4, p5 );
        }

        ~ScopeGuardImpl5() throw() 
        {
            SafeExecute( *this );
        }

        void Execute()
        {
            fun_( p1_, p2_, p3_, p4_, p5_ );
        }

    protected:
        ScopeGuardImpl5( F fun, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5 ) :
             fun_( fun ), p1_( p1 ), p2_( p2 ), p3_( p3 ), p4_( p4 ), p5_( p5 )
        {}

        F fun_;
        const P1 p1_;
        const P2 p2_;
        const P3 p3_;
        const P4 p4_;
        const P5 p5_;
    };

    template < typename F, typename P1, typename P2, typename P3, typename P4, typename P5 >
    inline ScopeGuardImpl5< F, P1, P2, P3, P4, P5 > MakeGuard( F fun, P1 p1, P2 p2, P3 p3, P4 p4, P5 p5 )
    {
        return ScopeGuardImpl5< F, P1, P2, P3, P4, P5 >::MakeGuard( fun, p1, p2, p3, p4, p5 );
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// \class ObjScopeGuardImpl0
    /// \ingroup ExceptionGroup
    ///
    /// Implementation class for a class per-instance member function with no
    /// parameters.  ScopeGuard ignores any value returned from the call within
    /// the Execute function.
    ///
    /// This class has 3 standalone helper functions which create a ScopeGuard.
    /// One is MakeObjGuard, which is deprecated but provided for older code.
    /// The other two are MakeGuard overloads, one which takes a pointer to an
    /// object, and the other which takes a reference.
    ///
    ////////////////////////////////////////////////////////////////

    template <class Obj, typename MemFun>
    class ObjScopeGuardImpl0 : public ScopeGuardImplBase
    {
    public:
        static ObjScopeGuardImpl0<Obj, MemFun> MakeObjGuard(Obj& obj, MemFun memFun)
        {
            return ObjScopeGuardImpl0<Obj, MemFun>(obj, memFun);
        }

        ~ObjScopeGuardImpl0() throw() 
        {
            SafeExecute(*this);
        }

        void Execute() 
        {
            (obj_.*memFun_)();
        }

    protected:
        ObjScopeGuardImpl0(Obj& obj, MemFun memFun) : obj_(obj), memFun_(memFun) 
        {}

        Obj& obj_;
        MemFun memFun_;
    };

    template <class Obj, typename MemFun>
    inline ObjScopeGuardImpl0<Obj, MemFun> MakeObjGuard(Obj& obj, MemFun memFun)
    {
        return ObjScopeGuardImpl0<Obj, MemFun>::MakeObjGuard(obj, memFun);
    }

    template <typename Ret, class Obj1, class Obj2>
    inline ObjScopeGuardImpl0<Obj1,Ret(Obj2::*)()> MakeGuard(Ret(Obj2::*memFun)(), Obj1 &obj) 
    {
      return ObjScopeGuardImpl0<Obj1,Ret(Obj2::*)()>::MakeObjGuard(obj,memFun);
    }

    template <typename Ret, class Obj1, class Obj2>
    inline ObjScopeGuardImpl0<Obj1,Ret(Obj2::*)()> MakeGuard(Ret(Obj2::*memFun)(), Obj1 *obj) 
    {
      return ObjScopeGuardImpl0<Obj1,Ret(Obj2::*)()>::MakeObjGuard(*obj,memFun);
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// \class ObjScopeGuardImpl1
    /// \ingroup ExceptionGroup
    ///
    /// Implementation class for a class per-instance member function with one
    /// parameter.  The parameter is copied by value - use ::Loki::ByRef if you
    /// must use a reference instead.  ScopeGuard ignores any value returned
    /// from the call within the Execute function.
    ///
    /// This class has 3 standalone helper functions which create a ScopeGuard.
    /// One is MakeObjGuard, which is deprecated but provided for older code.
    /// The other two are MakeGuard overloads, one which takes a pointer to an
    /// object, and the other which takes a reference.
    ///
    ////////////////////////////////////////////////////////////////

    template <class Obj, typename MemFun, typename P1>
    class ObjScopeGuardImpl1 : public ScopeGuardImplBase
    {
    public:
        static ObjScopeGuardImpl1<Obj, MemFun, P1> MakeObjGuard(Obj& obj, MemFun memFun, P1 p1)
        {
            return ObjScopeGuardImpl1<Obj, MemFun, P1>(obj, memFun, p1);
        }

        ~ObjScopeGuardImpl1() throw() 
        {
            SafeExecute(*this);
        }

        void Execute() 
        {
            (obj_.*memFun_)(p1_);
        }

    protected:
        ObjScopeGuardImpl1(Obj& obj, MemFun memFun, P1 p1) : obj_(obj), memFun_(memFun), p1_(p1) 
        {}
        
        Obj& obj_;
        MemFun memFun_;
        const P1 p1_;
    };

    template <class Obj, typename MemFun, typename P1>
    inline ObjScopeGuardImpl1<Obj, MemFun, P1> MakeObjGuard(Obj& obj, MemFun memFun, P1 p1)
    {
        return ObjScopeGuardImpl1<Obj, MemFun, P1>::MakeObjGuard(obj, memFun, p1);
    }

    template <typename Ret, class Obj1, class Obj2, typename P1a, typename P1b>
    inline ObjScopeGuardImpl1<Obj1,Ret(Obj2::*)(P1a),P1b> MakeGuard(Ret(Obj2::*memFun)(P1a), Obj1 &obj, P1b p1) 
    {
      return ObjScopeGuardImpl1<Obj1,Ret(Obj2::*)(P1a),P1b>::MakeObjGuard(obj,memFun,p1);
    }

    template <typename Ret, class Obj1, class Obj2, typename P1a, typename P1b>
    inline ObjScopeGuardImpl1<Obj1,Ret(Obj2::*)(P1a),P1b> MakeGuard(Ret(Obj2::*memFun)(P1a), Obj1 *obj, P1b p1) 
    {
      return ObjScopeGuardImpl1<Obj1,Ret(Obj2::*)(P1a),P1b>::MakeObjGuard(*obj,memFun,p1);
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// \class ObjScopeGuardImpl2
    /// \ingroup ExceptionGroup
    ///
    /// Implementation class for a class per-instance member function with two
    /// parameters.  Each parameter is copied by value - use ::Loki::ByRef if you
    /// must use a reference instead.  ScopeGuard ignores any value returned
    /// from the call within the Execute function.
    ///
    /// This class has 3 standalone helper functions which create a ScopeGuard.
    /// One is MakeObjGuard, which is deprecated but provided for older code.
    /// The other two are MakeGuard overloads, one which takes a pointer to an
    /// object, and the other which takes a reference.
    ///
    ////////////////////////////////////////////////////////////////

    template <class Obj, typename MemFun, typename P1, typename P2>
    class ObjScopeGuardImpl2 : public ScopeGuardImplBase
    {
    public:
        static ObjScopeGuardImpl2<Obj, MemFun, P1, P2> MakeObjGuard(Obj& obj, MemFun memFun, P1 p1, P2 p2)
        {
            return ObjScopeGuardImpl2<Obj, MemFun, P1, P2>(obj, memFun, p1, p2);
        }

        ~ObjScopeGuardImpl2() throw() 
        {
            SafeExecute(*this);
        }

        void Execute() 
        {
            (obj_.*memFun_)(p1_, p2_);
        }

    protected:
        ObjScopeGuardImpl2(Obj& obj, MemFun memFun, P1 p1, P2 p2) : obj_(obj), memFun_(memFun), p1_(p1), p2_(p2) 
        {}

        Obj& obj_;
        MemFun memFun_;
        const P1 p1_;
        const P2 p2_;
    };

    template <class Obj, typename MemFun, typename P1, typename P2>
    inline ObjScopeGuardImpl2<Obj, MemFun, P1, P2> MakeObjGuard(Obj& obj, MemFun memFun, P1 p1, P2 p2)
    {
        return ObjScopeGuardImpl2<Obj, MemFun, P1, P2>::MakeObjGuard(obj, memFun, p1, p2);
    }

    template <typename Ret, class Obj1, class Obj2, typename P1a, typename P1b, typename P2a, typename P2b>
    inline ObjScopeGuardImpl2<Obj1,Ret(Obj2::*)(P1a,P2a),P1b,P2b> MakeGuard(Ret(Obj2::*memFun)(P1a,P2a), Obj1 &obj, P1b p1, P2b p2) 
    {
      return ObjScopeGuardImpl2<Obj1,Ret(Obj2::*)(P1a,P2a),P1b,P2b>::MakeObjGuard(obj,memFun,p1,p2);
    }

    template <typename Ret, class Obj1, class Obj2, typename P1a, typename P1b, typename P2a, typename P2b>
    inline ObjScopeGuardImpl2<Obj1,Ret(Obj2::*)(P1a,P2a),P1b,P2b> MakeGuard(Ret(Obj2::*memFun)(P1a,P2a), Obj1 *obj, P1b p1, P2b p2) 
    {
      return ObjScopeGuardImpl2<Obj1,Ret(Obj2::*)(P1a,P2a),P1b,P2b>::MakeObjGuard(*obj,memFun,p1,p2);
    }

    ////////////////////////////////////////////////////////////////
    ///
    /// \class ObjScopeGuardImpl3
    /// \ingroup ExceptionGroup
    ///
    /// Implementation class for a class per-instance member function with three
    /// parameters.  Each parameter is copied by value - use ::Loki::ByRef if you
    /// must use a reference instead.  ScopeGuard ignores any value returned
    /// from the call within the Execute function.
    ///
    /// This class has 3 standalone helper functions which create a ScopeGuard.
    /// One is MakeObjGuard, which is deprecated but provided for older code.
    /// The other two are MakeGuard overloads, one which takes a pointer to an
    /// object, and the other which takes a reference.
    ///
    ////////////////////////////////////////////////////////////////

    template < class Obj, typename MemFun, typename P1, typename P2, typename P3 >
    class ObjScopeGuardImpl3 : public ScopeGuardImplBase
    {
    public:
        static ObjScopeGuardImpl3< Obj, MemFun, P1, P2, P3 > MakeObjGuard(
            Obj & obj, MemFun memFun, P1 p1, P2 p2, P3 p3 )
        {
            return ObjScopeGuardImpl3< Obj, MemFun, P1, P2, P3 >( obj, memFun, p1, p2, p3 );
        }

        ~ObjScopeGuardImpl3() throw() 
        {
            SafeExecute( *this );
        }

        void Execute() 
        {
            ( obj_.*memFun_ )( p1_, p2_, p3_ );
        }

    protected:
        ObjScopeGuardImpl3( Obj & obj, MemFun memFun, P1 p1, P2 p2, P3 p3 ) :
             obj_( obj ), memFun_( memFun ), p1_( p1 ), p2_( p2 ), p3_( p3 )
        {}

        Obj& obj_;
        MemFun memFun_;
        const P1 p1_;
        const P2 p2_;
        const P3 p3_;
    };

    template < class Obj, typename MemFun, typename P1, typename P2, typename P3 >
    inline ObjScopeGuardImpl3< Obj, MemFun, P1, P2, P3 > MakeObjGuard(
        Obj & obj, MemFun memFun, P1 p1, P2 p2, P3 p3 )
    {
        return ObjScopeGuardImpl3< Obj, MemFun, P1, P2, P3 >::MakeObjGuard(
            obj, memFun, p1, p2, p3 );
    }

    template < typename Ret, class Obj1, class Obj2, typename P1a, typename P1b,
        typename P2a, typename P2b, typename P3a, typename P3b >
    inline ObjScopeGuardImpl3< Obj1, Ret( Obj2::* )( P1a, P2a, P3a ), P1b, P2b, P3b >
        MakeGuard( Ret( Obj2::*memFun )( P1a, P2a, P3a ), Obj1 & obj, P1b p1, P2b p2, P3b p3 )
    {
      return ObjScopeGuardImpl3< Obj1, Ret( Obj2::* )( P1a, P2a, P3a ), P1b, P2b, P3b >
          ::MakeObjGuard( obj, memFun, p1, p2, p3 );
    }

    template < typename Ret, class Obj1, class Obj2, typename P1a, typename P1b,
        typename P2a, typename P2b, typename P3a, typename P3b >
    inline ObjScopeGuardImpl3< Obj1, Ret( Obj2::* )( P1a, P2a, P3a ), P1b, P2b, P3b >
        MakeGuard( Ret( Obj2::*memFun )( P1a, P2a, P3a ), Obj1 * obj, P1b p1, P2b p2, P3b p3 )
    {
      return ObjScopeGuardImpl3< Obj1, Ret( Obj2::* )( P1a, P2a, P3a ), P1b, P2b, P3b >
          ::MakeObjGuard( *obj, memFun, p1, p2, p3 );
    }

} // namespace Loki

#define LOKI_CONCATENATE_DIRECT(s1, s2)  s1##s2
#define LOKI_CONCATENATE(s1, s2)         LOKI_CONCATENATE_DIRECT(s1, s2)
#define LOKI_ANONYMOUS_VARIABLE(str)     LOKI_CONCATENATE(str, __LINE__)

#define LOKI_ON_BLOCK_EXIT      ::Loki::ScopeGuard LOKI_ANONYMOUS_VARIABLE(scopeGuard) = ::Loki::MakeGuard
#define LOKI_ON_BLOCK_EXIT_OBJ  ::Loki::ScopeGuard LOKI_ANONYMOUS_VARIABLE(scopeGuard) = ::Loki::MakeObjGuard

#endif // end file guardian

