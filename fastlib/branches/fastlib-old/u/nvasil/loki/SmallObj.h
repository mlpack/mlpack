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
#ifndef LOKI_SMALLOBJ_INC_
#define LOKI_SMALLOBJ_INC_

// $Id: SmallObj.h 806 2007-02-03 00:01:52Z rich_sposato $


#include "LokiExport.h"
#include "Threads.h"
#include "Singleton.h"
#include <cstddef>
#include <new> // needed for std::nothrow_t parameter.

#ifndef LOKI_DEFAULT_CHUNK_SIZE
#define LOKI_DEFAULT_CHUNK_SIZE 4096
#endif

#ifndef LOKI_MAX_SMALL_OBJECT_SIZE
#define LOKI_MAX_SMALL_OBJECT_SIZE 256
#endif

#ifndef LOKI_DEFAULT_OBJECT_ALIGNMENT
#define LOKI_DEFAULT_OBJECT_ALIGNMENT 4
#endif

#ifndef LOKI_DEFAULT_SMALLOBJ_LIFETIME
#define LOKI_DEFAULT_SMALLOBJ_LIFETIME ::Loki::LongevityLifetime::DieAsSmallObjectParent
#endif

#if defined(LOKI_SMALL_OBJECT_USE_NEW_ARRAY) && defined(_MSC_VER)
#pragma message("Don't define LOKI_SMALL_OBJECT_USE_NEW_ARRAY when using a Microsoft compiler to prevent memory leaks.")
#pragma message("now calling '#undef LOKI_SMALL_OBJECT_USE_NEW_ARRAY'")
#undef LOKI_SMALL_OBJECT_USE_NEW_ARRAY
#endif

///  \defgroup  SmallObjectGroup Small objects
///
///  \defgroup  SmallObjectGroupInternal Internals
///  \ingroup   SmallObjectGroup

namespace Loki
{
    namespace LongevityLifetime
    {
        /** @struct DieAsSmallObjectParent
            @ingroup SmallObjectGroup
            Lifetime policy to manage lifetime dependencies of 
            SmallObject base and child classes.
            The Base class should have this lifetime
        */
        template <class T>
        struct DieAsSmallObjectParent  : DieLast<T> {};

        /** @struct DieAsSmallObjectChild
            @ingroup SmallObjectGroup
            Lifetime policy to manage lifetime dependencies of 
            SmallObject base and child classes.
            The Child class should have this lifetime
        */
        template <class T>
        struct DieAsSmallObjectChild  : DieDirectlyBeforeLast<T> {};

    } 

    class FixedAllocator;

    /** @class SmallObjAllocator
        @ingroup SmallObjectGroupInternal
     Manages pool of fixed-size allocators.
     Designed to be a non-templated base class of AllocatorSingleton so that
     implementation details can be safely hidden in the source code file.
     */
    class LOKI_EXPORT SmallObjAllocator
    {
    protected:
        /** The only available constructor needs certain parameters in order to
         initialize all the FixedAllocator's.  This throws only if
         @param pageSize # of bytes in a page of memory.
         @param maxObjectSize Max # of bytes which this may allocate.
         @param objectAlignSize # of bytes between alignment boundaries.
         */
        SmallObjAllocator( std::size_t pageSize, std::size_t maxObjectSize,
            std::size_t objectAlignSize );

        /** Destructor releases all blocks, all Chunks, and FixedAllocator's.
         Any outstanding blocks are unavailable, and should not be used after
         this destructor is called.  The destructor is deliberately non-virtual
         because it is protected, not public.
         */
        ~SmallObjAllocator( void );

    public:
        /** Allocates a block of memory of requested size.  Complexity is often
         constant-time, but might be O(C) where C is the number of Chunks in a
         FixedAllocator. 

         @par Exception Safety Level
         Provides either strong-exception safety, or no-throw exception-safety
         level depending upon doThrow parameter.  The reason it provides two
         levels of exception safety is because it is used by both the nothrow
         and throwing new operators.  The underlying implementation will never
         throw of its own accord, but this can decide to throw if it does not
         allocate.  The only exception it should emit is std::bad_alloc.

         @par Allocation Failure
         If it does not allocate, it will call TrimExcessMemory and attempt to
         allocate again, before it decides to throw or return NULL.  Many
         allocators loop through several new_handler functions, and terminate
         if they can not allocate, but not this one.  It only makes one attempt
         using its own implementation of the new_handler, and then returns NULL
         or throws so that the program can decide what to do at a higher level.
         (Side note: Even though the C++ Standard allows allocators and
         new_handlers to terminate if they fail, the Loki allocator does not do
         that since that policy is not polite to a host program.)

         @param size # of bytes needed for allocation.
         @param doThrow True if this should throw if unable to allocate, false
          if it should provide no-throw exception safety level.
         @return NULL if nothing allocated and doThrow is false.  Else the
          pointer to an available block of memory.
         */
        void * Allocate( std::size_t size, bool doThrow );

        /** Deallocates a block of memory at a given place and of a specific
        size.  Complexity is almost always constant-time, and is O(C) only if
        it has to search for which Chunk deallocates.  This never throws.
         */
        void Deallocate( void * p, std::size_t size );

        /** Deallocates a block of memory at a given place but of unknown size
        size.  Complexity is O(F + C) where F is the count of FixedAllocator's
        in the pool, and C is the number of Chunks in all FixedAllocator's.  This
        does not throw exceptions.  This overloaded version of Deallocate is
        called by the nothow delete operator - which is called when the nothrow
        new operator is used, but a constructor throws an exception.
         */
        void Deallocate( void * p );

        /// Returns max # of bytes which this can allocate.
        inline std::size_t GetMaxObjectSize() const
        { return maxSmallObjectSize_; }

        /// Returns # of bytes between allocation boundaries.
        inline std::size_t GetAlignment() const { return objectAlignSize_; }

        /** Releases empty Chunks from memory.  Complexity is O(F + C) where F
        is the count of FixedAllocator's in the pool, and C is the number of
        Chunks in all FixedAllocator's.  This will never throw.  This is called
        by AllocatorSingleto::ClearExtraMemory, the new_handler function for
        Loki's allocator, and is called internally when an allocation fails.
        @return True if any memory released, or false if none released.
         */
        bool TrimExcessMemory( void );

        /** Returns true if anything in implementation is corrupt.  Complexity
         is O(F + C + B) where F is the count of FixedAllocator's in the pool,
         C is the number of Chunks in all FixedAllocator's, and B is the number
         of blocks in all Chunks.  If it determines any data is corrupted, this
         will return true in release version, but assert in debug version at
         the line where it detects the corrupted data.  If it does not detect
         any corrupted data, it returns false.
         */
        bool IsCorrupt( void ) const;

    private:
        /// Default-constructor is not implemented.
        SmallObjAllocator( void );
        /// Copy-constructor is not implemented.
        SmallObjAllocator( const SmallObjAllocator & );
        /// Copy-assignment operator is not implemented.
        SmallObjAllocator & operator = ( const SmallObjAllocator & );

        /// Pointer to array of fixed-size allocators.
        Loki::FixedAllocator * pool_;

        /// Largest object size supported by allocators.
        const std::size_t maxSmallObjectSize_;

        /// Size of alignment boundaries.
        const std::size_t objectAlignSize_;
    };


    /** @class AllocatorSingleton
        @ingroup SmallObjectGroupInternal
     This template class is derived from
     SmallObjAllocator in order to pass template arguments into it, and still
     have a default constructor for the singleton.  Each instance is a unique
     combination of all the template parameters, and hence is singleton only 
     with respect to those parameters.  The template parameters have default
     values and the class has typedefs identical to both SmallObject and
     SmallValueObject so that this class can be used directly instead of going
     through SmallObject or SmallValueObject.  That design feature allows
     clients to use the new_handler without having the name of the new_handler
     function show up in classes derived from SmallObject or SmallValueObject.
     Thus, the only functions in the allocator which show up in SmallObject or
     SmallValueObject inheritance hierarchies are the new and delete
     operators.
    */
    template
    <
        template <class, class> class ThreadingModel = LOKI_DEFAULT_THREADING_NO_OBJ_LEVEL,
        std::size_t chunkSize = LOKI_DEFAULT_CHUNK_SIZE,
        std::size_t maxSmallObjectSize = LOKI_MAX_SMALL_OBJECT_SIZE,
        std::size_t objectAlignSize = LOKI_DEFAULT_OBJECT_ALIGNMENT,
        template <class> class LifetimePolicy = LOKI_DEFAULT_SMALLOBJ_LIFETIME,
        class MutexPolicy = LOKI_DEFAULT_MUTEX
    >
    class AllocatorSingleton : public SmallObjAllocator
    {
    public:

        /// Defines type of allocator.
        typedef AllocatorSingleton< ThreadingModel, chunkSize,
            maxSmallObjectSize, objectAlignSize, LifetimePolicy > MyAllocator;

        /// Defines type for thread-safety locking mechanism.
        typedef ThreadingModel< MyAllocator, MutexPolicy > MyThreadingModel;

        /// Defines singleton made from allocator.
        typedef Loki::SingletonHolder< MyAllocator, Loki::CreateStatic,
            LifetimePolicy, ThreadingModel > MyAllocatorSingleton;

        /// Returns reference to the singleton.
        inline static AllocatorSingleton & Instance( void )
        {
            return MyAllocatorSingleton::Instance();
        }

        /// The default constructor is not meant to be called directly.
        inline AllocatorSingleton() :
            SmallObjAllocator( chunkSize, maxSmallObjectSize, objectAlignSize )
            {}

        /// The destructor is not meant to be called directly.
        inline ~AllocatorSingleton( void ) {}

        /** Clears any excess memory used by the allocator.  Complexity is
         O(F + C) where F is the count of FixedAllocator's in the pool, and C
         is the number of Chunks in all FixedAllocator's.  This never throws.
         @note This function can be used as a new_handler when Loki and other
         memory allocators can no longer allocate.  Although the C++ Standard
         allows new_handler functions to terminate the program when they can
         not release any memory, this will not do so.
         */
        static void ClearExtraMemory( void );

        /** Returns true if anything in implementation is corrupt.  Complexity
         is O(F + C + B) where F is the count of FixedAllocator's in the pool,
         C is the number of Chunks in all FixedAllocator's, and B is the number
         of blocks in all Chunks.  If it determines any data is corrupted, this
         will return true in release version, but assert in debug version at
         the line where it detects the corrupted data.  If it does not detect
         any corrupted data, it returns false.
         */
        static bool IsCorrupted( void );

    private:
        /// Copy-constructor is not implemented.
        AllocatorSingleton( const AllocatorSingleton & );
        /// Copy-assignment operator is not implemented.
        AllocatorSingleton & operator = ( const AllocatorSingleton & );
    };

    template
    <
        template <class, class> class T,
        std::size_t C,
        std::size_t M,
        std::size_t O,
        template <class> class L,
        class X
    >
    void AllocatorSingleton< T, C, M, O, L, X >::ClearExtraMemory( void )
    {
        typename MyThreadingModel::Lock lock;
        (void)lock; // get rid of warning
        Instance().TrimExcessMemory();
    }

    template
    <
        template <class, class> class T,
        std::size_t C,
        std::size_t M,
        std::size_t O,
        template <class> class L,
        class X
    >
    bool AllocatorSingleton< T, C, M, O, L, X >::IsCorrupted( void )
    {
        typename MyThreadingModel::Lock lock;
        (void)lock; // get rid of warning
        return Instance().IsCorrupt();
    }

    /** This standalone function provides the longevity level for Small-Object
     Allocators which use the Loki::SingletonWithLongevity policy.  The
     SingletonWithLongevity class can find this function through argument-
     dependent lookup.

     @par Longevity Levels
     No Small-Object Allocator depends on any other Small-Object allocator, so
     this does not need to calculate dependency levels among allocators, and
     it returns just a constant.  All allocators must live longer than the
     objects which use the allocators, it must return a longevity level higher
     than any such object.
     */
    template
    <
        template <class, class> class T,
        std::size_t C,
        std::size_t M,
        std::size_t O,
        template <class> class L,
        class X
    >
    inline unsigned int GetLongevity(
        AllocatorSingleton< T, C, M, O, L, X > * )
    {
        // Returns highest possible value.
        return 0xFFFFFFFF;
    }


    /** @class SmallObjectBase
        @ingroup SmallObjectGroup
     Base class for small object allocation classes.
     The shared implementation of the new and delete operators are here instead
     of being duplicated in both SmallObject or SmallValueObject, later just 
     called Small-Objects.  This class is not meant to be used directly by clients, 
     or derived from by clients. Class has no data members so compilers can 
     use Empty-Base-Optimization.

     @par ThreadingModel
     This class doesn't support ObjectLevelLockable policy for ThreadingModel.
     The allocator is a singleton, so a per-instance mutex is not necessary.
     Nor is using ObjectLevelLockable recommended with SingletonHolder since
     the SingletonHolder::MakeInstance function requires a mutex that exists
     prior to when the object is created - which is not possible if the mutex
     is inside the object, such as required for ObjectLevelLockable.  If you
     attempt to use ObjectLevelLockable, the compiler will emit errors because
     it can't use the default constructor in ObjectLevelLockable.  If you need
     a thread-safe allocator, use the ClassLevelLockable policy.

     @par Lifetime Policy
     
     The SmallObjectBase template needs a lifetime policy because it owns
     a singleton of SmallObjAllocator which does all the low level functions. 
     When using a Small-Object in combination with the SingletonHolder template
     you have to choose two lifetimes, that of the Small-Object and that of
     the singleton. The rule is: The Small-Object lifetime must be greater than
     the lifetime of the singleton hosting the Small-Object. Violating this rule
     results in a crash on exit, because the hosting singleton tries to delete
     the Small-Object which is then already destroyed. 
     
     The lifetime policies recommended for use with Small-Objects hosted 
     by a SingletonHolder template are 
         - LongevityLifetime::DieAsSmallObjectParent / LongevityLifetime::DieAsSmallObjectChild
         - SingletonWithLongevity
         - FollowIntoDeath (not supported by MSVC 7.1)
         - NoDestroy
     
     The default lifetime of Small-Objects is 
     LongevityLifetime::DieAsSmallObjectParent to
     insure that memory is not released before a object with the lifetime
     LongevityLifetime::DieAsSmallObjectChild using that
     memory is destroyed. The LongevityLifetime::DieAsSmallObjectParent
     lifetime has the highest possible value of a SetLongevity lifetime, so
     you can use it in combination with your own lifetime not having also
     the highest possible value.
     
     The DefaultLifetime and PhoenixSingleton policies are *not* recommended 
     since they can cause the allocator to be destroyed and release memory 
     for singletons hosting a object which inherit from either SmallObject
     or SmallValueObject.  
     
     @par Lifetime usage
    
        - LongevityLifetime: The Small-Object has 
          LongevityLifetime::DieAsSmallObjectParent policy and the Singleton
          hosting the Small-Object has LongevityLifetime::DieAsSmallObjectChild. 
          The child lifetime has a hard coded SetLongevity lifetime which is 
          shorter than the lifetime of the parent, thus the child dies 
          before the parent.
         
        - Both Small-Object and Singleton use SingletonWithLongevity policy.
          The longevity level for the singleton must be lower than that for the
          Small-Object. This is why the AllocatorSingleton's GetLongevity function 
          returns the highest value.
         
        - FollowIntoDeath lifetime: The Small-Object has 
          FollowIntoDeath::With<LIFETIME>::AsMasterLiftime
          policy and the Singleton has 
          FollowIntoDeath::AfterMaster<MASTERSINGLETON>::IsDestroyed policy,
          where you could choose the LIFETIME. 
        
        - Both Small-Object and Singleton use NoDestroy policy. 
          Since neither is ever destroyed, the destruction order does not matter.
          Note: you will get memory leaks!
         
        - The Small-Object has NoDestroy policy but the Singleton has
          SingletonWithLongevity policy. Note: you will get memory leaks!
         
     
     You should *not* use NoDestroy for the singleton, and then use
     SingletonWithLongevity for the Small-Object. 
     
     @par Examples:
     
     - test/SmallObj/SmallSingleton.cpp
     - test/Singleton/Dependencies.cpp
     */
    template
    <
        template <class, class> class ThreadingModel,
        std::size_t chunkSize,
        std::size_t maxSmallObjectSize,
        std::size_t objectAlignSize,
        template <class> class LifetimePolicy,
        class MutexPolicy
    >
    class SmallObjectBase
    {

#if (LOKI_MAX_SMALL_OBJECT_SIZE != 0) && (LOKI_DEFAULT_CHUNK_SIZE != 0) && (LOKI_DEFAULT_OBJECT_ALIGNMENT != 0)

    public:        
        /// Defines type of allocator singleton, must be public 
        /// to handle singleton lifetime dependencies.
        typedef AllocatorSingleton< ThreadingModel, chunkSize,
            maxSmallObjectSize, objectAlignSize, LifetimePolicy > ObjAllocatorSingleton;
    
    private:

        /// Defines type for thread-safety locking mechanism.
        typedef ThreadingModel< ObjAllocatorSingleton, MutexPolicy > MyThreadingModel;

        /// Use singleton defined in AllocatorSingleton.
        typedef typename ObjAllocatorSingleton::MyAllocatorSingleton MyAllocatorSingleton;
        
    public:

        /// Throwing single-object new throws bad_alloc when allocation fails.
#ifdef _MSC_VER
        /// @note MSVC complains about non-empty exception specification lists.
        static void * operator new ( std::size_t size )
#else
        static void * operator new ( std::size_t size ) throw ( std::bad_alloc )
#endif
        {
            typename MyThreadingModel::Lock lock;
            (void)lock; // get rid of warning
            return MyAllocatorSingleton::Instance().Allocate( size, true );
        }

        /// Non-throwing single-object new returns NULL if allocation fails.
        static void * operator new ( std::size_t size, const std::nothrow_t & ) throw ()
        {
            typename MyThreadingModel::Lock lock;
            (void)lock; // get rid of warning
            return MyAllocatorSingleton::Instance().Allocate( size, false );
        }

        /// Placement single-object new merely calls global placement new.
        inline static void * operator new ( std::size_t size, void * place )
        {
            return ::operator new( size, place );
        }

        /// Single-object delete.
        static void operator delete ( void * p, std::size_t size ) throw ()
        {
            typename MyThreadingModel::Lock lock;
            (void)lock; // get rid of warning
            MyAllocatorSingleton::Instance().Deallocate( p, size );
        }

        /** Non-throwing single-object delete is only called when nothrow
         new operator is used, and the constructor throws an exception.
         */
        static void operator delete ( void * p, const std::nothrow_t & ) throw()
        {
            typename MyThreadingModel::Lock lock;
            (void)lock; // get rid of warning
            MyAllocatorSingleton::Instance().Deallocate( p );
        }

        /// Placement single-object delete merely calls global placement delete.
        inline static void operator delete ( void * p, void * place )
        {
            ::operator delete ( p, place );
        }

#ifdef LOKI_SMALL_OBJECT_USE_NEW_ARRAY

        /// Throwing array-object new throws bad_alloc when allocation fails.
#ifdef _MSC_VER
        /// @note MSVC complains about non-empty exception specification lists.
        static void * operator new [] ( std::size_t size )
#else
        static void * operator new [] ( std::size_t size )
            throw ( std::bad_alloc )
#endif
        {
            typename MyThreadingModel::Lock lock;
            (void)lock; // get rid of warning
            return MyAllocatorSingleton::Instance().Allocate( size, true );
        }

        /// Non-throwing array-object new returns NULL if allocation fails.
        static void * operator new [] ( std::size_t size,
            const std::nothrow_t & ) throw ()
        {
            typename MyThreadingModel::Lock lock;
            (void)lock; // get rid of warning
            return MyAllocatorSingleton::Instance().Allocate( size, false );
        }

        /// Placement array-object new merely calls global placement new.
        inline static void * operator new [] ( std::size_t size, void * place )
        {
            return ::operator new( size, place );
        }

        /// Array-object delete.
        static void operator delete [] ( void * p, std::size_t size ) throw ()
        {
            typename MyThreadingModel::Lock lock;
            (void)lock; // get rid of warning
            MyAllocatorSingleton::Instance().Deallocate( p, size );
        }

        /** Non-throwing array-object delete is only called when nothrow
         new operator is used, and the constructor throws an exception.
         */
        static void operator delete [] ( void * p,
            const std::nothrow_t & ) throw()
        {
            typename MyThreadingModel::Lock lock;
            (void)lock; // get rid of warning
            MyAllocatorSingleton::Instance().Deallocate( p );
        }

        /// Placement array-object delete merely calls global placement delete.
        inline static void operator delete [] ( void * p, void * place )
        {
            ::operator delete ( p, place );
        }
#endif  // #if use new array functions.

#endif  // #if default template parameters are not zero

    protected:
        inline SmallObjectBase( void ) {}
        inline SmallObjectBase( const SmallObjectBase & ) {}
        inline SmallObjectBase & operator = ( const SmallObjectBase & )
        { return *this; }
        inline ~SmallObjectBase() {}
    }; // end class SmallObjectBase


    /** @class SmallObject
        @ingroup SmallObjectGroup
     SmallObject Base class for polymorphic small objects, offers fast
     allocations & deallocations.  Destructor is virtual and public.  Default
     constructor is trivial.   Copy-constructor and copy-assignment operator are
     not implemented since polymorphic classes almost always disable those
     operations.  Class has no data members so compilers can use
     Empty-Base-Optimization.
     */
    template
    <
        template <class, class> class ThreadingModel = LOKI_DEFAULT_THREADING_NO_OBJ_LEVEL,
        std::size_t chunkSize = LOKI_DEFAULT_CHUNK_SIZE,
        std::size_t maxSmallObjectSize = LOKI_MAX_SMALL_OBJECT_SIZE,
        std::size_t objectAlignSize = LOKI_DEFAULT_OBJECT_ALIGNMENT,
        template <class> class LifetimePolicy = LOKI_DEFAULT_SMALLOBJ_LIFETIME,
        class MutexPolicy = LOKI_DEFAULT_MUTEX
    >
    class SmallObject : public SmallObjectBase< ThreadingModel, chunkSize,
            maxSmallObjectSize, objectAlignSize, LifetimePolicy, MutexPolicy >
    {

    public:
        virtual ~SmallObject() {}
    protected:
        inline SmallObject( void ) {}

    private:
        /// Copy-constructor is not implemented.
        SmallObject( const SmallObject & );
        /// Copy-assignment operator is not implemented.
        SmallObject & operator = ( const SmallObject & );
    }; // end class SmallObject


    /** @class SmallValueObject
        @ingroup SmallObjectGroup
     SmallValueObject Base class for small objects with value-type
     semantics - offers fast allocations & deallocations.  Destructor is
     non-virtual, inline, and protected to prevent unintentional destruction
     through base class.  Default constructor is trivial.   Copy-constructor
     and copy-assignment operator are trivial since value-types almost always
     need those operations.  Class has no data members so compilers can use
     Empty-Base-Optimization.
     */
    template
    <
        template <class, class> class ThreadingModel = LOKI_DEFAULT_THREADING_NO_OBJ_LEVEL,
        std::size_t chunkSize = LOKI_DEFAULT_CHUNK_SIZE,
        std::size_t maxSmallObjectSize = LOKI_MAX_SMALL_OBJECT_SIZE,
        std::size_t objectAlignSize = LOKI_DEFAULT_OBJECT_ALIGNMENT,
        template <class> class LifetimePolicy = LOKI_DEFAULT_SMALLOBJ_LIFETIME,
        class MutexPolicy = LOKI_DEFAULT_MUTEX
    >
    class SmallValueObject : public SmallObjectBase< ThreadingModel, chunkSize,
            maxSmallObjectSize, objectAlignSize, LifetimePolicy, MutexPolicy >
    {
    protected:
        inline SmallValueObject( void ) {}
        inline SmallValueObject( const SmallValueObject & ) {}
        inline SmallValueObject & operator = ( const SmallValueObject & )
        { return *this; }
        inline ~SmallValueObject() {}
    }; // end class SmallValueObject

} // namespace Loki

#endif // end file guardian

