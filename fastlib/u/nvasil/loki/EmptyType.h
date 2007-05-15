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
#ifndef LOKI_EMPTYTYPE_INC_
#define LOKI_EMPTYTYPE_INC_

// $Id: EmptyType.h 751 2006-10-17 19:50:37Z syntheticpp $


namespace Loki
{
////////////////////////////////////////////////////////////////////////////////
// class EmptyType
// Used as a class type that doesn't hold anything
// Useful as a strawman class
////////////////////////////////////////////////////////////////////////////////

    class EmptyType {};
    
    
    inline bool operator==(const EmptyType&, const EmptyType&)
    {
        return true;
    }   

    inline bool operator<(const EmptyType&, const EmptyType&)
    {
        return false;
    }
    
    inline bool operator>(const EmptyType&, const EmptyType&)
    {
        return false;
    }
}

#endif // end file guardian

