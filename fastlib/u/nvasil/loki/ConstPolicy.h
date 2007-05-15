////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2006 Richard Sposato
// Copyright (c) 2006 Peter Kümmel
// Permission to use, copy, modify, distribute and sell this software for any 
//     purpose is hereby granted without fee, provided that the above copyright 
//     notice appear in all copies and that both that copyright notice and this 
//     permission notice appear in supporting documentation.
// The authors make no representations about the 
//     suitability of this software for any purpose. It is provided "as is" 
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////
#ifndef LOKI_CONST_POLICY_INC_
#define LOKI_CONST_POLICY_INC_

// $Id: ConstPolicy.h 769 2006-10-26 10:58:19Z syntheticpp $


namespace Loki
{

////////////////////////////////////////////////////////////////////////////////
/// @note These policy classes are used in LockingPtr and SmartPtr to define
///  how const is propagated from the pointee.
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
///  \class DontPropagateConst
///
///  \ingroup ConstGroup
///  Don't propagate constness of pointed or referred object.
////////////////////////////////////////////////////////////////////////////////

    template< class T >
    struct DontPropagateConst
    {
        typedef T Type;
    };

////////////////////////////////////////////////////////////////////////////////
///  \class PropagateConst
///
///  \ingroup ConstGroup
///  Propagate constness of pointed or referred object.
////////////////////////////////////////////////////////////////////////////////

    template< class T >
    struct PropagateConst
    {
        typedef const T Type;
    };

// default will not break existing code
#ifndef LOKI_DEFAULT_CONSTNESS
#define LOKI_DEFAULT_CONSTNESS ::Loki::DontPropagateConst
#endif

} // end namespace Loki

#endif // end file guardian
