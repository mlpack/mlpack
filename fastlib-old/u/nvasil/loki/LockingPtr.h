////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2001 by Andrei Alexandrescu
// This code is from the article:
//     "Generic<Programming>: volatile — Multithreaded Programmer’s Best Friend
//     Volatile-Correctness or How to Have Your Compiler Detect Race Conditions
//     for You" by Alexandrescu, Andrei.
//     Published in the February 2001 issue of the C/C++ Users Journal.
//     http://www.cuj.com/documents/s=7998/cujcexp1902alexandr/
// Prepared for Loki library by Richard Sposato
////////////////////////////////////////////////////////////////////////////////
#ifndef LOKI_LOCKING_PTR_INC_
#define LOKI_LOCKING_PTR_INC_

// $Id: LockingPtr.h 748 2006-10-17 19:49:08Z syntheticpp $


#include <loki/ConstPolicy.h>

namespace Loki
{
    /** @class LockingPtr
     Locks a volatile object and casts away volatility so that the object
     can be safely used in a single-threaded region of code.
     Original version of LockingPtr had only one template - for the shared
     object, but not the mutex type.  This version allows users to specify a
     the mutex type as a LockingPolicy class.  The only requirements for a
     LockingPolicy class are to provide Lock and Unlock methods.
     */
    template < typename SharedObject, typename LockingPolicy = LOKI_DEFAULT_MUTEX, 
               template<class> class ConstPolicy = LOKI_DEFAULT_CONSTNESS >
    class LockingPtr
    {
    public:

        typedef typename ConstPolicy<SharedObject>::Type ConstOrNotType;

        /** Constructor locks mutex associated with an object.
         @param object Reference to object.
         @param mutex Mutex used to control thread access to object.
         */
        LockingPtr( volatile ConstOrNotType & object, LockingPolicy & mutex )
           : pObject_( const_cast< SharedObject * >( &object ) ),
            pMutex_( &mutex )
        {
            mutex.Lock();
        }

        typedef typename std::pair<volatile ConstOrNotType *, LockingPolicy *> Pair;

        /** Constructor locks mutex associated with an object.
         @param lockpair a std::pair of pointers to the object and the mutex
         */
        LockingPtr( Pair lockpair )
           : pObject_( const_cast< SharedObject * >( lockpair.first ) ),
            pMutex_( lockpair.second )
        {
            lockpair.second->Lock();
        }

        /// Destructor unlocks the mutex.
        ~LockingPtr()
        {
            pMutex_->Unlock();
        }

        /// Star-operator dereferences pointer.
        ConstOrNotType & operator * ()
        {
            return *pObject_;
        }

        /// Point-operator returns pointer to object.
        ConstOrNotType * operator -> ()
        {
            return pObject_;
        }

    private:

        /// Default constructor is not implemented.
        LockingPtr();

        /// Copy-constructor is not implemented.
        LockingPtr( const LockingPtr & );

        /// Copy-assignment-operator is not implemented.
        LockingPtr & operator = ( const LockingPtr & );

        /// Pointer to the shared object.
        ConstOrNotType * pObject_;

        /// Pointer to the mutex.
        LockingPolicy * pMutex_;

    }; // end class LockingPtr

} // namespace Loki

#endif // end file guardian

