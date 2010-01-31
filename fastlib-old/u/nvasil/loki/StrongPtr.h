////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2006 Rich Sposato
// The copyright on this file is protected under the terms of the MIT license.
//
// Permission to use, copy, modify, distribute and sell this software for any 
//     purpose is hereby granted without fee, provided that the above copyright 
//     notice appear in all copies and that both that copyright notice and this 
//     permission notice appear in supporting documentation.
// The author makes no representations about the 
//     suitability of this software for any purpose. It is provided "as is" 
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////
#ifndef LOKI_STRONG_PTR_INC_
#define LOKI_STRONG_PTR_INC_

// $Id: StrongPtr.h 807 2007-02-25 12:49:19Z syntheticpp $


#include <loki/SmartPtr.h>
#if defined (LOKI_OBJECT_LEVEL_THREADING) || defined (LOKI_CLASS_LEVEL_THREADING)
    #include <loki/Threads.h>
#endif


////////////////////////////////////////////////////////////////////////////////
///
///  \par Terminology
///   These terms are used within this file's comments.
///   -# StrongPtr : Class used to implement both strong and weak pointers. The
///      second template parameter determines if a StrongPtr is weak or strong.
///   -# Strong pointer : A pointer that claims ownership of a shared object.
///      When the last strong copointer dies, the object is destroyed even if
///      there are weak copointers.
///   -# Weak pointer : A pointer that does not own the shared object it points
///       to.  It only destroys the shared object if there no strong copointers
///       exist when it dies.
///   -# Copointers : All the pointers that refer to the same shared object.
///      The copointers must have the same ownership policy, but the other
///      policies may be different.
///   -# Pointee : The shared object.
///
///  \par OwnershipPolicy
///   The ownership policy has the pointer to the actual object, and it also
///   keeps track of the strong and weak copointers so that it can know if any
///   strong copointers remain.  The plain pointer it maintains is stored as a
///   void pointer, which allows the ownership policy classes to be monolithic
///   classes instead of template classes.  As monolithic classes, they reduce
///   amount of code-bloat.
///
///  \par Writing Your Own OwnershipPolicy
///   If you write your own policy, you must implement these 12 functions:
///   -# explicit YourPolicy( bool strong )
///   -# YourPolicy( void * p, bool strong )
///   -# YourPolicy( const YourPolicy & rhs, bool strong )
///   -# bool Release( bool strong )
///   -# void Increment( bool strong )
///   -# bool Decrement( bool strong )
///   -# bool HasStrongPointer( void ) const
///   -# void Swap( YourPolicy & rhs )
///   -# void SetPointer( void * p )
///   -# void ZapPointer( void )
///   -# void * GetPointer( void ) const
///   -# void * & GetPointerRef( void ) const
///   It is strongly recommended that all 12 of these functions be protected
///   instead of public.  These two functions are optional for single-threaded
///   policies, but required for multi-threaded policies:
///   -# void Lock( void ) const
///   -# void Unlock( void ) const
///   This function is entirely optional:
///   -# bool Merge( TwoRefLinks & rhs )
///
///  \par DeletePolicy
///   The delete policy provides a mechanism to destroy an object and a default
///   value for an uninitialized pointer.  You can override this policy with
///   your own when using the Singleton, NullObject, or Prototype design
///   patterns.
///
///  \par Writing Your Own DeletePolicy
///   If you write your own policy, you must implement these 3 functions:
///   -# void static Delete( const P * p )
///   -# static P * Default( void )
///   -# void Swap( YourResetPolicy & )
///
///  \par ResetPolicy
///   A reset policy tells the ReleaseAll and ResetAll functions whether they
///   should release or reset the StrongPtr copointers.  These functions do
///   not affect just one StrongPtr, but all copointers.  That is unlike
///   SmartPtr where the Release and Reset functions only affect 1 SmartPtr,
///   and leave all copointers untouched.  A useful trick you can do with the
///   ResetPolicy is to not allow reset when a strong pointer exists, and then
///   use the NoCheck policy for all strong pointers.  The reset policy
///   guarantees the strong pointers always have a valid pointee, so checking
///   is not required; but weak pointers may still require checking.
///
///  \par Writing Your Own ResetPolicy
///   If you write your own policy, you must implement these 2 functions:
///   -# bool OnReleaseAll( bool ) const
///   -# bool OnResetAll( bool ) const
///   The bool parameter means that this was called with a strong pointer or
///   one of its copointers is strong.  The return value means the pointer
///   can be reset or released.
///
///  \defgroup  StrongPointerOwnershipGroup StrongPtr Ownership policies
///  \ingroup   SmartPointerGroup
///  \defgroup  StrongPointerDeleteGroup Delete policies
///  \ingroup   SmartPointerGroup
///  \defgroup  StrongPointerResetGroup Reset policies
///  \ingroup   SmartPointerGroup
////////////////////////////////////////////////////////////////////////////////


namespace Loki
{


////////////////////////////////////////////////////////////////////////////////
///  \class DeleteUsingFree
///
///  \ingroup  StrongPointerDeleteGroup 
///  Implementation of the DeletePolicy used by StrongPtr.  Uses explicit call
///   to T's destructor followed by call to free.  This policy is useful for
///   managing the lifetime of pointers to structs returned by C functions.
////////////////////////////////////////////////////////////////////////////////

template < class P >
class DeleteUsingFree
{
public:
    inline void static Delete( const P * p )
    {
        if ( 0 != p )
        {
            p->~P();
            ::free( p );
        }
    }

    /// Provides default value to initialize the pointer
    inline static P * Default( void )
    {
        return 0;
    }

    inline void Swap( DeleteUsingFree & ) {}
};

////////////////////////////////////////////////////////////////////////////////
///  \class DeleteNothing
///
///  \ingroup  StrongPointerDeleteGroup 
///  Implementation of the DeletePolicy used by StrongPtr.  This will never
///   delete anything.  You can use this policy with pointers to an undefined
///   type or a pure interface class with a protected destructor.
////////////////////////////////////////////////////////////////////////////////

template < class P >
class DeleteNothing
{
public:
    inline static void Delete( const P * )
    {
        // Do nothing at all!
    }

    inline static P * Default( void )
    {
        return 0;
    }

    inline void Swap( DeleteNothing & ) {}
};

////////////////////////////////////////////////////////////////////////////////
///  \class DeleteSingle
///
///  \ingroup  StrongPointerDeleteGroup 
///  Implementation of the DeletePolicy used by StrongPtr.  This deletes just
///   one shared object.  This is the default class for the DeletePolicy.
////////////////////////////////////////////////////////////////////////////////

template < class P >
class DeleteSingle
{
public:
    inline static void Delete( const P * p )
    {
        /** @note If you see an error message about a negative subscript, that
         means your are attempting to use Loki to delete an incomplete type.
         Please don't use this policy with incomplete types; you may want to
         use DeleteNothing instead.
         */
        typedef char Type_Must_Be_Defined[ sizeof(P) ? 1 : -1 ];
        delete p;
    }

    inline static P * Default( void )
    {
        return 0;
    }

    inline void Swap( DeleteSingle & ) {}
};

////////////////////////////////////////////////////////////////////////////////
///  \class DeleteArray
///
///  \ingroup  StrongPointerDeleteGroup 
///  Implementation of the DeletePolicy used by StrongPtr.  This deletes an
///   array of shared objects.
////////////////////////////////////////////////////////////////////////////////

template < class P >
class DeleteArray
{
public:
    inline static void Delete( const P * p )
    {
        /** @note If you see an error message about a negative subscript, that
         means your are attempting to use Loki to delete an incomplete type.
         Please don't use this policy with incomplete types; you may want to
         use DeleteNothing instead.
         */
        typedef char Type_Must_Be_Defined[ sizeof(P) ? 1 : -1 ];
        delete [] p;
    }

    inline static P * Default( void )
    {
        return 0;
    }

    inline void Swap( DeleteArray & ) {}
};

////////////////////////////////////////////////////////////////////////////////
///  \class CantResetWithStrong
///
///  \ingroup  StrongPointerResetGroup 
///  Implementation of the ResetPolicy used by StrongPtr.  This is the default
///   ResetPolicy for StrongPtr.  It forbids reset and release only if a strong
///   copointer exists.
////////////////////////////////////////////////////////////////////////////////

template < class P >
struct CantResetWithStrong
{
    inline bool OnReleaseAll( bool hasStrongPtr ) const
    {
        return ! hasStrongPtr;
    }

    inline bool OnResetAll( bool hasStrongPtr ) const
    {
        return ! hasStrongPtr;
    }
};

////////////////////////////////////////////////////////////////////////////////
///  \class AllowReset
///
///  \ingroup  StrongPointerResetGroup 
///  Implementation of the ResetPolicy used by StrongPtr.  It allows reset and
///   release under any circumstance.
////////////////////////////////////////////////////////////////////////////////

template < class P >
struct AllowReset
{
    inline bool OnReleaseAll( bool ) const
    {
        return true;
    }
    inline bool OnResetAll( bool ) const
    {
        return true;
    }
};

////////////////////////////////////////////////////////////////////////////////
///  \class NeverReset
///
///  \ingroup  StrongPointerResetGroup 
///  Implementation of the ResetPolicy used by StrongPtr.  It forbids reset and
///   release under any circumstance.
////////////////////////////////////////////////////////////////////////////////

template < class P >
struct NeverReset
{
    inline bool OnReleaseAll( bool ) const
    {
        return false;
    }
    inline bool OnResetAll( bool ) const
    {
        return false;
    }
};

// ----------------------------------------------------------------------------

namespace Private
{

////////////////////////////////////////////////////////////////////////////////
///  \class TwoRefCountInfo
///
///  \ingroup  StrongPointerOwnershipGroup
///   Implementation detail for reference counting strong and weak pointers.
///   It maintains a void pointer and 2 reference counts.  Since it is just a
///   class for managing implementation details, it is not intended to be used
///   directly - which is why it is in a private namespace.  Each instance is a
///   shared resource for all copointers, and there should be only one of these
///   for each set of copointers.  This class is small, trivial, and inline.
////////////////////////////////////////////////////////////////////////////////

class LOKI_EXPORT TwoRefCountInfo
{
public:

    inline explicit TwoRefCountInfo( bool strong )
        : m_pointer( 0 )
        , m_strongCount( strong ? 1 : 0 )
        , m_weakCount( strong ? 0 : 1 )
    {
    }

    inline TwoRefCountInfo( void * p, bool strong )
        : m_pointer( p )
        , m_strongCount( strong ? 1 : 0 )
        , m_weakCount( strong ? 0 : 1 )
    {
    }

    inline ~TwoRefCountInfo( void )
    {
        assert( 0 == m_strongCount );
        assert( 0 == m_weakCount );
    }

    inline bool HasStrongPointer( void ) const
    {
        return ( 0 < m_strongCount );
    }

    inline bool HasWeakPointer( void ) const
    {
        return ( 0 < m_weakCount );
    }

    inline void IncStrongCount( void )
    {
        ++m_strongCount;
    }

    inline void IncWeakCount( void )
    {
        ++m_weakCount;
    }

    inline void DecStrongCount( void )
    {
        assert( 0 < m_strongCount );
        --m_strongCount;
    }

    inline void DecWeakCount( void )
    {
        assert( 0 < m_weakCount );
        --m_weakCount;
    }

    inline void ZapPointer( void )
    {
        m_pointer = 0;
    }

    void SetPointer( void * p )
    {
        m_pointer = p;
    }

    inline void * GetPointer( void ) const
    {
        return m_pointer;
    }

    inline void * & GetPointerRef( void ) const
    {
        return const_cast< void * & >( m_pointer );
    }

private:
    /// Copy-constructor not implemented.
    TwoRefCountInfo( const TwoRefCountInfo & );
    /// Copy-assignment operator not implemented.
    TwoRefCountInfo & operator = ( const TwoRefCountInfo & );

    void * m_pointer;
    unsigned int m_strongCount;
    unsigned int m_weakCount;
};

////////////////////////////////////////////////////////////////////////////////
///  \class LockableTwoRefCountInfo
///
///  \ingroup  StrongPointerOwnershipGroup
///   Implementation detail for thread-safe reference counting for strong and
///   weak pointers.  It uses TwoRefCountInfo to manage the pointer and counts.
///   All this does is provide a thread safety mechanism.  Since it is just a
///   class for managing implementation details, it is not intended to be used
///   directly - which is why it is in a private namespace.  Each instance is a
///   shared resource for all copointers, and there should be only one of these
///   for each set of copointers.  This class is small, trivial, and inline.
///
///  \note This class is not designed for use with a single-threaded model.
///   Tests using a single-threaded model will not run properly, but tests in a
///   multi-threaded model with either class-level-locking or object-level-locking
///   do run properly.
////////////////////////////////////////////////////////////////////////////////

#if defined (LOKI_OBJECT_LEVEL_THREADING) || defined (LOKI_CLASS_LEVEL_THREADING)

class LOKI_EXPORT LockableTwoRefCountInfo
    : private Loki::Private::TwoRefCountInfo
{
public:

    inline explicit LockableTwoRefCountInfo( bool strong )
        : TwoRefCountInfo( strong )
        , m_Mutex()
    {
    }

    LockableTwoRefCountInfo( void * p, bool strong )
        : TwoRefCountInfo( p, strong )
        , m_Mutex()
    {
    }

    inline ~LockableTwoRefCountInfo( void )
    {
    }

    inline void Lock( void ) const
    {
        m_Mutex.Lock();
    }

    inline void Unlock( void ) const
    {
        m_Mutex.Unlock();
    }

    inline bool HasStrongPointer( void ) const
    {
        m_Mutex.Lock();
        const bool has = TwoRefCountInfo::HasStrongPointer();
        m_Mutex.Unlock();
        return has;
    }

    inline bool HasWeakPointer( void ) const
    {
        m_Mutex.Lock();
        const bool has = TwoRefCountInfo::HasWeakPointer();
        m_Mutex.Unlock();
        return has;
    }

    inline void IncStrongCount( void )
    {
        m_Mutex.Lock();
        TwoRefCountInfo::IncStrongCount();
        m_Mutex.Unlock();
    }

    inline void IncWeakCount( void )
    {
        m_Mutex.Lock();
        TwoRefCountInfo::IncWeakCount();
        m_Mutex.Unlock();
    }

    inline void DecStrongCount( void )
    {
        m_Mutex.Lock();
        TwoRefCountInfo::DecStrongCount();
        m_Mutex.Unlock();
    }

    inline void DecWeakCount( void )
    {
        m_Mutex.Lock();
        TwoRefCountInfo::DecWeakCount();
        m_Mutex.Unlock();
    }

    inline void ZapPointer( void )
    {
        m_Mutex.Lock();
        TwoRefCountInfo::ZapPointer();
        m_Mutex.Unlock();
    }

    void SetPointer( void * p )
    {
        m_Mutex.Lock();
        TwoRefCountInfo::SetPointer( p );
        m_Mutex.Unlock();
    }

    inline void * GetPointer( void ) const
    {
        return TwoRefCountInfo::GetPointer();
    }

    inline void * & GetPointerRef( void ) const
    {
        return TwoRefCountInfo::GetPointerRef();
    }

private:
    /// Default constructor is not available.
    LockableTwoRefCountInfo( void );
    /// Copy constructor is not available.
    LockableTwoRefCountInfo( const LockableTwoRefCountInfo & );
    /// Copy-assignment operator is not available.
    LockableTwoRefCountInfo & operator = ( const LockableTwoRefCountInfo & );

    mutable LOKI_DEFAULT_MUTEX m_Mutex;
};

#endif // if object-level-locking or class-level-locking

} // end namespace Private

////////////////////////////////////////////////////////////////////////////////
///  \class TwoRefCounts
///
///  \ingroup  StrongPointerOwnershipGroup
///   This implementation of StrongPtr's OwnershipPolicy uses a pointer to a
///   shared instance of TwoRefCountInfo.  This is the default policy for
///   OwnershipPolicy.  Some functions are trivial enough to be inline, while
///   others are implemented elsewhere.  It is not thread safe, and is intended
///   for single-threaded environments.
////////////////////////////////////////////////////////////////////////////////

class LOKI_EXPORT TwoRefCounts
{
protected:

    explicit TwoRefCounts( bool strong );

    TwoRefCounts( const void * p, bool strong );

    TwoRefCounts( const TwoRefCounts & rhs, bool strong ) :
        m_counts( rhs.m_counts )
    {
        Increment( strong );
    }

    inline bool Release( bool strong )
    {
        return Decrement( strong );
    }

    void Increment( bool strong );

    bool Decrement( bool strong );

    bool HasStrongPointer( void ) const
    {
        return m_counts->HasStrongPointer();
    }

    void Swap( TwoRefCounts & rhs );

    void SetPointer( void * p )
    {
        m_counts->SetPointer( p );
    }

    void ZapPointer( void );

    inline void * & GetPointerRef( void ) const
    {
        return m_counts->GetPointerRef();
    }

    inline void * GetPointer( void ) const
    {
        return m_counts->GetPointer();
    }

private:
    TwoRefCounts( void );
    TwoRefCounts & operator = ( const TwoRefCounts & );

    /// Pointer to all shared data.
    Loki::Private::TwoRefCountInfo * m_counts;
};

////////////////////////////////////////////////////////////////////////////////
///  \class LockableTwoRefCounts
///
///  \ingroup  StrongPointerOwnershipGroup
///   This implementation of StrongPtr's OwnershipPolicy uses a pointer to a
///   shared instance of LockableTwoRefCountInfo.  It behaves very similarly to
///   TwoRefCounts, except that it provides thread-safety.  Some functions are
///   trivial enough to be inline, while others are implemented elsewhere.
///
///  \note This class is not designed for use with a single-threaded model.
///   Tests using a single-threaded model will not run properly, but tests in a
///   multi-threaded model with either class-level-locking or object-level-locking
///   do run properly.
////////////////////////////////////////////////////////////////////////////////

#if defined (LOKI_OBJECT_LEVEL_THREADING) || defined (LOKI_CLASS_LEVEL_THREADING)

class LOKI_EXPORT LockableTwoRefCounts
{
    typedef SmallValueObject< ::Loki::ClassLevelLockable > ThreadSafePointerAllocator;

protected:

    explicit LockableTwoRefCounts( bool strong )
        : m_counts( NULL )
    {
        void * temp = ThreadSafePointerAllocator::operator new(
            sizeof(Loki::Private::LockableTwoRefCountInfo) );
#ifdef DO_EXTRA_LOKI_TESTS
        assert( temp != 0 );
#endif
        m_counts = new ( temp ) Loki::Private::LockableTwoRefCountInfo( strong );
    }

    LockableTwoRefCounts( const void * p, bool strong )
        : m_counts( NULL )
    {
        void * temp = ThreadSafePointerAllocator::operator new(
            sizeof(Loki::Private::LockableTwoRefCountInfo) );
#ifdef DO_EXTRA_LOKI_TESTS
        assert( temp != 0 );
#endif
        void * p2 = const_cast< void * >( p );
        m_counts = new ( temp )
            Loki::Private::LockableTwoRefCountInfo( p2, strong );
    }

    LockableTwoRefCounts( const LockableTwoRefCounts & rhs, bool strong ) :
        m_counts( rhs.m_counts )
    {
        Increment( strong );
    }

    inline void Lock( void ) const
    {
        m_counts->Lock();
    }

    inline void Unlock( void ) const
    {
        m_counts->Unlock();
    }

    inline bool Release( bool strong )
    {
        return Decrement( strong );
    }

    void Increment( bool strong )
    {
        if ( strong )
        {
            m_counts->IncStrongCount();
        }
        else
        {
            m_counts->IncWeakCount();
        }
    }

    bool Decrement( bool strong )
    {
        if ( strong )
        {
            m_counts->DecStrongCount();
        }
        else
        {
            m_counts->DecWeakCount();
        }
        return !m_counts->HasStrongPointer();
    }

    bool HasStrongPointer( void ) const
    {
        return m_counts->HasStrongPointer();
    }

    void Swap( LockableTwoRefCounts & rhs )
    {
        std::swap( m_counts, rhs.m_counts );
    }

    void SetPointer( void * p )
    {
        m_counts->SetPointer( p );
    }

    void ZapPointer( void )
    {
#ifdef DO_EXTRA_LOKI_TESTS
        assert( !m_counts->HasStrongPointer() );
#endif
        if ( m_counts->HasWeakPointer() )
        {
            m_counts->ZapPointer();
        }
        else
        {
            ThreadSafePointerAllocator::operator delete ( m_counts,
                sizeof(Loki::Private::LockableTwoRefCountInfo) );
            m_counts = NULL;
        }
    }

    inline void * GetPointer( void ) const
    {
        return m_counts->GetPointer();
    }

    inline void * & GetPointerRef( void ) const
    {
        return m_counts->GetPointerRef();
    }

private:
    LockableTwoRefCounts( void );
    LockableTwoRefCounts & operator = ( const LockableTwoRefCounts & );

    /// Pointer to all shared data.
    Loki::Private::LockableTwoRefCountInfo * m_counts;
};

#endif // if object-level-locking or class-level-locking

////////////////////////////////////////////////////////////////////////////////
///  \class TwoRefLinks
///
///  \ingroup  StrongPointerOwnershipGroup
///   This implementation of StrongPtr's OwnershipPolicy uses a doubly-linked
///   cycle of copointers to a shared object. Some functions are trivial enough
///   to be inline, while others are implemented in elsewhere.  It is not thread
///   safe, and is intended for single-threaded environments.
////////////////////////////////////////////////////////////////////////////////

class LOKI_EXPORT TwoRefLinks
{
protected:

    inline explicit TwoRefLinks( bool strong )
        : m_pointer( 0 )
        , m_strong( strong )
    {
        m_prev = m_next = this;
    }

    TwoRefLinks( const void * p, bool strong );

    TwoRefLinks( const TwoRefLinks & rhs, bool strong );

    bool Release( bool strong );

    void Swap( TwoRefLinks & rhs );

    bool Merge( TwoRefLinks & rhs );

    bool HasStrongPointer( void ) const;

    inline void ZapPointer( void )
    {
        ZapAllNodes();
    }

    void SetPointer( void * p );

    inline void * GetPointer( void ) const
    {
        return m_pointer;
    }

    inline void * & GetPointerRef( void ) const
    {
        return const_cast< void * & >( m_pointer );
    }

private:
    static unsigned int CountPrevCycle( const TwoRefLinks * pThis );
    static unsigned int CountNextCycle( const TwoRefLinks * pThis );

    /// Not implemented.
    TwoRefLinks( void );
    /// Not implemented.
    TwoRefLinks & operator = ( const TwoRefLinks & );

    bool HasPrevNode( const TwoRefLinks * p ) const;
    bool HasNextNode( const TwoRefLinks * p ) const;
    bool AllNodesHaveSamePointer( void ) const;
    void ZapAllNodes( void );

    void * m_pointer;
    mutable TwoRefLinks * m_prev;
    mutable TwoRefLinks * m_next;
    const bool m_strong;
};

////////////////////////////////////////////////////////////////////////////////
///  \class StrongPtr
///
///  \ingroup SmartPointerGroup 
///
///  \param Strong           default = true,
///  \param OwnershipPolicy  default = TwoRefCounts,
///  \param ConversionPolicy default = DisallowConversion,
///  \param CheckingPolicy   default = AssertCheck,
///  \param ResetPolicy      default = CantResetWithStrong,
///  \param DeletePolicy     default = DeleteSingle
///  \param ConstnessPolicy  default = LOKI_DEFAULT_CONSTNESS
////////////////////////////////////////////////////////////////////////////////

template
<
    typename T,
    bool Strong = true,
    class OwnershipPolicy = Loki::TwoRefCounts,
    class ConversionPolicy = Loki::DisallowConversion,
    template < class > class CheckingPolicy = Loki::AssertCheck,
    template < class > class ResetPolicy = Loki::CantResetWithStrong,
    template < class > class DeletePolicy = Loki::DeleteSingle,
    template < class > class ConstnessPolicy = LOKI_DEFAULT_CONSTNESS
>
class StrongPtr
    : public OwnershipPolicy
    , public ConversionPolicy
    , public CheckingPolicy< T * >
    , public ResetPolicy< T >
    , public DeletePolicy< T >
{
    typedef ConversionPolicy CP;
    typedef CheckingPolicy< T * > KP;
    typedef ResetPolicy< T > RP;
    typedef DeletePolicy< T > DP;

public:

    typedef OwnershipPolicy OP;

    typedef T * StoredType;    // the type of the pointer
    typedef T * PointerType;   // type returned by operator->
    typedef T & ReferenceType; // type returned by operator*

    typedef typename ConstnessPolicy< T >::Type * ConstPointerType;
    typedef typename ConstnessPolicy< T >::Type & ConstReferenceType;

private:
    struct NeverMatched {};

#ifdef LOKI_SMARTPTR_CONVERSION_CONSTRUCTOR_POLICY
    typedef typename Select< CP::allow, const StoredType&, NeverMatched>::Result ImplicitArg;
    typedef typename Select<!CP::allow, const StoredType&, NeverMatched>::Result ExplicitArg;
#else
    typedef const StoredType& ImplicitArg;
    typedef typename Select<false, const StoredType&, NeverMatched>::Result ExplicitArg;
#endif

public:

    StrongPtr( void ) : OP( Strong )
    {
        KP::OnDefault( GetPointer() );
    }

    explicit StrongPtr( ExplicitArg p ) : OP( p, Strong )
    {
        KP::OnInit( GetPointer() );
    }

    StrongPtr( ImplicitArg p ) : OP( p, Strong )
    {
        KP::OnInit( GetPointer() );
    }

    StrongPtr( const StrongPtr & rhs )
        : OP( rhs, Strong ), CP( rhs ), KP( rhs ), DP( rhs )
    {
    }

    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    StrongPtr(
        const StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & rhs )
        : OP( rhs, Strong )
    {
    }

    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    StrongPtr(
        StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & rhs )
        : OP( rhs, Strong )
    {
    }

    StrongPtr( RefToValue< StrongPtr > rhs )
        : OP( rhs, Strong ), KP( rhs ), CP( rhs ), DP( rhs )
    {
    }

    operator RefToValue< StrongPtr >( void )
    {
        return RefToValue< StrongPtr >( *this );
    }

    StrongPtr & operator = ( const StrongPtr & rhs )
    {
        if ( GetPointer() != rhs.GetPointer() )
        {
            StrongPtr temp( rhs );
            temp.Swap( *this );
        }
        return *this;
    }

    StrongPtr & operator = ( T * p )
    {
        if ( GetPointer() != p )
        {
            StrongPtr temp( p );
            Swap( temp );
        }
        return *this;
    }

    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    StrongPtr & operator = (
        const StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & rhs )
    {
        if ( !rhs.Equals( GetPointer() ) )
        {
            StrongPtr temp( rhs );
            temp.Swap( *this );
        }
        return *this;
    }

    bool IsStrong( void ) const
    {
        return Strong;
    }

    void Swap( StrongPtr & rhs )
    {
        OP::Swap( rhs );
        CP::Swap( rhs );
        KP::Swap( rhs );
        DP::Swap( rhs );
    }

    ~StrongPtr()
    {
        if ( OP::Release( Strong ) )
        {
            // Must zap the pointer before deleteing the object. Otherwise a
            // cycle of weak pointers will lead to recursion, which leads to
            // to deleting the shared object multiple times, which leads to
            // undefined behavior.  Therefore, this must get pointer before
            // zapping it, and then delete the temp pointer.
            T * p = GetPointer();
            if ( p != 0 )
            {
                OP::ZapPointer();
                DP::Delete( p );
            }
        }
    }

#ifdef LOKI_ENABLE_FRIEND_TEMPLATE_TEMPLATE_PARAMETER_WORKAROUND

    // old non standard in class definition of friends
    friend bool ReleaseAll( StrongPtr & sp,
        typename StrongPtr::StoredType & p )
    {
        if ( !sp.RP::OnReleaseAll( sp.IsStrong() || sp.OP::HasStrongPointer() ) )
        {
            return false;
        }
        p = sp.GetPointer();
        sp.OP::SetPointer( sp.DP::Default() );
        return true;
    }

    friend bool ResetAll( StrongPtr & sp,
        typename StrongPtr::StoredType p )
    {
        if ( sp.OP::GetPointer() == p )
        {
            return true;
        }
        if ( !sp.RP::OnResetAll( sp.IsStrong() || sp.OP::HasStrongPointer() ) )
        {
            return false;
        }
        sp.DP::Delete( sp.GetPointer() );
        sp.OP::SetPointer( p );
        return true;
    }

#else
  
    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    friend bool ReleaseAll( StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & sp,
        typename StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 >::StoredType & p );
 

    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    friend bool ResetAll( StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & sp,
        typename StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 >::StoredType p );

#endif


    /** Merges ownership of two StrongPtr's that point to same shared object
      but are not copointers.  Requires Merge function in OwnershipPolicy.
      \return True for success, false if not pointer to same object.
     */
    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    bool Merge( StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & rhs )
    {
        if ( OP::GetPointer() != rhs.OP::GetPointer() )
        {
            return false;
        }
        return OP::Merge( rhs );
    }

    /** Locks StrongPtr so other threads can't affect pointer.  Requires the
     OwnershipPolicy to have Lock function.
     */
    void Lock( void )
    {
        OP::Lock();
    }

    /** Unlocks StrongPtr so other threads can affect pointer.  Requires the
     OwnershipPolicy to have Unlock function.
     */
    void Unlock( void )
    {
        OP::Unlock();
    }

    PointerType operator -> ()
    {
        KP::OnDereference( GetPointer() );
        return GetPointer();
    }

    ConstPointerType operator -> () const
    {
        KP::OnDereference( GetPointer() );
        return GetPointer();
    }

    ReferenceType operator * ()
    {
        KP::OnDereference( GetPointer() );
        return * GetPointer();
    }

    ConstReferenceType operator * () const
    {
        KP::OnDereference( GetPointer() );
        return * GetPointer();
    }

    /// Helper function which can be called to avoid exposing GetPointer function.
    template < class T1 >
    bool Equals( const T1 * p ) const
    {
        return ( GetPointer() == p );
    }

    /// Helper function which can be called to avoid exposing GetPointer function.
    template < class T1 >
    bool LessThan( const T1 * p ) const
    {
        return ( GetPointer() < p );
    }

    /// Helper function which can be called to avoid exposing GetPointer function.
    template < class T1 >
    bool GreaterThan( const T1 * p ) const
    {
        return ( GetPointer() > p );
    }

    /// Equality comparison operator is templated to handle ambiguity.
    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    bool operator == (
        const StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & rhs ) const
    {
        return ( rhs.Equals( GetPointer() ) );
    }

    /// Inequality comparison operator is templated to handle ambiguity.
    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    bool operator != (
        const StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & rhs ) const
    {
        return !( rhs.Equals( GetPointer() ) );
    }

    /// Less-than comparison operator is templated to handle ambiguity.
    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    bool operator < (
        const StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & rhs ) const
    {
        return ( rhs.GreaterThan( GetPointer() ) );
    }

    /// Greater-than comparison operator is templated to handle ambiguity.
    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    inline bool operator > (
        const StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & rhs ) const
    {
        return ( rhs.LessThan( GetPointer() ) );
    }

    /// Less-than-or-equal-to operator is templated to handle ambiguity.
    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    inline bool operator <= (
        const StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & rhs ) const
    {
        return !( rhs.LessThan( GetPointer() ) );
    }

    /// Greater-than-or-equal-to operator is templated to handle ambiguity.
    template
    <
        typename T1,
        bool S1,
        class OP1,
        class CP1,
        template < class > class KP1,
        template < class > class RP1,
        template < class > class DP1,
        template < class > class CNP1
    >
    inline bool operator >= (
        const StrongPtr< T1, S1, OP1, CP1, KP1, RP1, DP1, CNP1 > & rhs ) const
    {
        return !( rhs.GreaterThan( GetPointer() ) );
    }

    inline bool operator ! () const // Enables "if ( !sp ) ..."
    {
        return ( 0 == OP::GetPointer() );
    }

protected:

    inline PointerType GetPointer( void )
    {
        return reinterpret_cast< PointerType >( OP::GetPointer() );
    }

    inline ConstPointerType GetPointer( void ) const
    {
        return reinterpret_cast< ConstPointerType >( OP::GetPointer() );
    }

private:

    inline ReferenceType GetPointerRef( void )
    {
        return reinterpret_cast< ReferenceType >( OP::GetPointerRef() );
    }

    inline ConstReferenceType GetPointerRef( void ) const
    {
        return reinterpret_cast< ConstReferenceType >( OP::GetPointerRef() );
    }

    // Helper for enabling 'if (sp)'
    struct Tester
    {
        Tester(int) {}
        void dummy() {}
    };
    
    typedef void (Tester::*unspecified_boolean_type_)();

    typedef typename Select< CP::allow, Tester, unspecified_boolean_type_ >::Result
        unspecified_boolean_type;

public:
    // enable 'if (sp)'
    operator unspecified_boolean_type() const
    {
        return !*this ? 0 : &Tester::dummy;
    }

private:
    // Helper for disallowing automatic conversion
    struct Insipid
    {
        Insipid(PointerType) {}
    };
    
    typedef typename Select< CP::allow, PointerType, Insipid >::Result
        AutomaticConversionResult;

public:        
    operator AutomaticConversionResult() const
    {
        return GetPointer();
    }

};

// ----------------------------------------------------------------------------

// friend functions

#ifndef LOKI_ENABLE_FRIEND_TEMPLATE_TEMPLATE_PARAMETER_WORKAROUND

template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
bool ReleaseAll( StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & sp,
                 typename StrongPtr< T, S, OP, CP, KP, RP, DP, CNP >::StoredType & p )
{
  if ( !sp.RP<T>::OnReleaseAll( sp.IsStrong() || sp.OP::HasStrongPointer() ) )
  {
    return false;
  }
  p = sp.GetPointer();
  sp.OP::SetPointer( sp.DP<T>::Default() );
  return true;
}

template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
bool ResetAll( StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & sp,
               typename StrongPtr< T, S, OP, CP, KP, RP, DP, CNP >::StoredType p )
{
  if ( sp.OP::GetPointer() == p )
  {
    return true;
  }
  if ( !sp.RP<T>::OnResetAll( sp.IsStrong() || sp.OP::HasStrongPointer() ) )
  {
    return false;
  }
  sp.DP<T>::Delete( sp.GetPointer() );
  sp.OP::SetPointer( p );
  return true;
}
#endif


// free comparison operators for class template StrongPtr

///  operator== for lhs = StrongPtr, rhs = raw pointer
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator == (
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & lhs, U * rhs )
{
    return ( lhs.Equals( rhs ) );
}

///  operator== for lhs = raw pointer, rhs = StrongPtr
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator == ( U * lhs,
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & rhs )
{
    return ( rhs.Equals( lhs ) );
}

///  operator!= for lhs = StrongPtr, rhs = raw pointer
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator != (
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & lhs, U * rhs )
{
    return !( lhs.Equals( rhs ) );
}

///  operator!= for lhs = raw pointer, rhs = StrongPtr
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator != ( U * lhs,
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & rhs )
{
    return !( rhs.Equals( lhs ) );
}

///  operator< for lhs = StrongPtr, rhs = raw pointer
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator < (
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & lhs, U * rhs )
{
    return ( lhs.LessThan( rhs ) );
}

///  operator< for lhs = raw pointer, rhs = StrongPtr
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator < ( U * lhs,
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & rhs )
{
    return ( rhs.GreaterThan( lhs ) );
}

//  operator> for lhs = StrongPtr, rhs = raw pointer
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator > (
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & lhs, U * rhs )
{
    return ( lhs.GreaterThan( rhs ) );
}

///  operator> for lhs = raw pointer, rhs = StrongPtr
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator > ( U * lhs,
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & rhs )
{
    return ( rhs.LessThan( lhs ) );
}

///  operator<= for lhs = StrongPtr, rhs = raw pointer
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator <= (
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & lhs, U * rhs )
{
    return !( lhs.GreaterThan( rhs ) );
}

///  operator<= for lhs = raw pointer, rhs = StrongPtr
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator <= ( U * lhs,
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & rhs )
{
    return !( rhs.LessThan( lhs ) );
}

///  operator>= for lhs = StrongPtr, rhs = raw pointer
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator >= (
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & lhs, U * rhs )
{
    return !( lhs.LessThan( rhs ) );
}

///  operator>= for lhs = raw pointer, rhs = StrongPtr
///  \ingroup SmartPointerGroup
template
<
    typename U,
    typename T,
    bool S,
    class OP,
    class CP,
    template < class > class KP,
    template < class > class RP,
    template < class > class DP,
    template < class > class CNP
>
inline bool operator >= ( U * lhs,
    const StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & rhs )
{
    return !( rhs.GreaterThan( lhs ) );
}

} // namespace Loki

namespace std
{
    ////////////////////////////////////////////////////////////////////////////////
    ///  specialization of std::less for StrongPtr
    ///  \ingroup SmartPointerGroup
    ////////////////////////////////////////////////////////////////////////////////
    template
    <
        typename T,
        bool S,
        class OP,
        class CP,
        template < class > class KP,
        template < class > class RP,
        template < class > class DP,
        template < class > class CNP
    >
    struct less< Loki::StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > >
        : public binary_function<
            Loki::StrongPtr< T, S, OP, CP, KP, RP, DP, CNP >,
            Loki::StrongPtr< T, S, OP, CP, KP, RP, DP, CNP >, bool >
    {
        bool operator () (
            const Loki::StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & lhs,
            const Loki::StrongPtr< T, S, OP, CP, KP, RP, DP, CNP > & rhs ) const
        {
            return ( lhs < rhs );
        }
    };
}

////////////////////////////////////////////////////////////////////////////////

#endif // end file guardian

