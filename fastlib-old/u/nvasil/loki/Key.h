////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2006 by Guillaume Chatelet
//
// Code covered by the MIT License
//
// Permission to use, copy, modify, distribute and sell this software for any 
// purpose is hereby granted without fee, provided that the above copyright 
// notice appear in all copies and that both that copyright notice and this 
// permission notice appear in supporting documentation.
//
// The authors make no representations about the suitability of this software
// for any purpose. It is provided "as is" without express or implied warranty.
//
// This code DOES NOT accompany the book:
// Alexandrescu, Andrei. "Modern C++ Design: Generic Programming and Design 
//     Patterns Applied". Copyright (c) 2001. Addison-Wesley.
//
////////////////////////////////////////////////////////////////////////////////
#ifndef LOKI_KEY_INC_
#define LOKI_KEY_INC_

// $Id: Key.h 771 2006-10-27 18:05:03Z clitte_bbt $


#include <loki/Factory.h>

namespace Loki
{

     template<
         class Factory,
         typename IdentifierType
     >
     class Key;

    template<class F, typename I>
    bool operator==(const Key<F, I> &k1, const Key<F, I> &k2);

    template<class F, typename I>
    bool operator<(const Key<F, I> &k1, const Key<F, I> &k2);
        

    /**
     * A Key class
     */
     template<
         class Factory,
         typename IdentifierType
     >
     class Key
     {
        typedef typename Factory::Parm1 Parm1;
        typedef typename Factory::Parm2 Parm2;
        typedef typename Factory::Parm3 Parm3;
        typedef typename Factory::Parm4 Parm4;
        typedef typename Factory::Parm5 Parm5;
        typedef typename Factory::Parm6 Parm6;
        typedef typename Factory::Parm7 Parm7;
        typedef typename Factory::Parm8 Parm8;
        typedef typename Factory::Parm9 Parm9;
        typedef typename Factory::Parm10 Parm10;
        typedef typename Factory::Parm11 Parm11;
        typedef typename Factory::Parm12 Parm12;
        typedef typename Factory::Parm13 Parm13;
        typedef typename Factory::Parm14 Parm14;
        typedef typename Factory::Parm15 Parm15;
     public:
        // member variables
        int count; // should be const, but constness prevent default copy ctor
        IdentifierType id;
        Parm1  p1;
        Parm2  p2;
        Parm3  p3;
        Parm4  p4;
        Parm5  p5;
        Parm6  p6;
        Parm7  p7;
        Parm8  p8;
        Parm9  p9;
        Parm10 p10;
        Parm11 p11;
        Parm12 p12;
        Parm13 p13;
        Parm14 p14;
        Parm15 p15;
        
        // member functions
        Key() : count(-1)
        {
        }

        Key(const IdentifierType& id) : count(0)
        {
            this->id = id;
        }
        
        Key(const IdentifierType& id,
				    Parm1 &p1) : count(1)
        {
            this->id = id;
            this->p1 = p1;
        }

        Key(const IdentifierType& id,
				    Parm1 &p1, Parm2 &p2) : count(2)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
        }

        Key(const IdentifierType& id,
				    Parm1 &p1, Parm2 &p2, Parm3 &p3) : count(3)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
        }

        Key(const IdentifierType& id,
				    Parm1 &p1, Parm2 &p2, Parm3 &p3, Parm4 &p4) : count(4)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
        }

        Key(const IdentifierType& id,
				    Parm1 &p1, Parm2 &p2, Parm3 &p3, Parm4 &p4, Parm5 &p5) : count(5)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
            this->p5 = p5;
        }

        Key(const IdentifierType& id,
				    Parm1 &p1, Parm2 &p2, Parm3 &p3, Parm4 &p4, Parm5 &p5,
				    Parm6 &p6) : count(6)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
            this->p5 = p5;
            this->p6 = p6;
        }

        Key(const IdentifierType& id,
				    Parm1 &p1, Parm2 &p2, Parm3 &p3, Parm4 &p4, Parm5 &p5,
				    Parm6 &p6, Parm7 &p7 ) : count(7)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
            this->p5 = p5;
            this->p6 = p6;
            this->p7 = p7;
        }

        Key(const IdentifierType& id,
				    Parm1 &p1, Parm2 &p2, Parm3 &p3, Parm4 &p4, Parm5 &p5,
				    Parm6 &p6, Parm7 &p7, Parm8 &p8) : count(8)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
            this->p5 = p5;
            this->p6 = p6;
            this->p7 = p7;
            this->p8 = p8;
        }

        Key(const IdentifierType& id,
				    Parm1 &p1, Parm2 &p2, Parm3 &p3, Parm4 &p4, Parm5 &p5,
				    Parm6 &p6, Parm7 &p7, Parm8 &p8, Parm9 &p9) : count(9)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
            this->p5 = p5;
            this->p6 = p6;
            this->p7 = p7;
            this->p8 = p8;
            this->p9 = p9;
        }
        
        Key(const IdentifierType& id,
				    Parm1 &p1, Parm2 &p2, Parm3 &p3, Parm4 &p4, Parm5 &p5,
				    Parm6 &p6, Parm7 &p7, Parm8 &p8, Parm9 &p9,Parm10 &p10) : count(10)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
            this->p5 = p5;
            this->p6 = p6;
            this->p7 = p7;
            this->p8 = p8;
            this->p9 = p9;
            this->p10 = p10;
        }

        Key(const IdentifierType& id,
				    Parm1  &p1, Parm2 &p2, Parm3 &p3, Parm4 &p4, Parm5  &p5,
				    Parm6  &p6, Parm7 &p7, Parm8 &p8, Parm9 &p9, Parm10 &p10,
				    Parm11 &p11) : count(11)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
            this->p5 = p5;
            this->p6 = p6;
            this->p7 = p7;
            this->p8 = p8;
            this->p9 = p9;
            this->p10 = p10;
            this->p11 = p11;
        }

        Key(const IdentifierType& id,
				    Parm1  &p1,  Parm2  &p2, Parm3 &p3, Parm4 &p4, Parm5  &p5,
				    Parm6  &p6,  Parm7  &p7, Parm8 &p8, Parm9 &p9, Parm10 &p10,
				    Parm11 &p11, Parm12 &p12) : count(12)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
            this->p5 = p5;
            this->p6 = p6;
            this->p7 = p7;
            this->p8 = p8;
            this->p9 = p9;
            this->p10 = p10;
            this->p11 = p11;
            this->p12 = p12;
        }

        Key(const IdentifierType& id,
				    Parm1  &p1,  Parm2  &p2,  Parm3  &p3, Parm4 &p4, Parm5  &p5,
				    Parm6  &p6,  Parm7  &p7,  Parm8  &p8, Parm9 &p9, Parm10 &p10,
				    Parm11 &p11, Parm12 &p12, Parm13 &p13) : count(13)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
            this->p5 = p5;
            this->p6 = p6;
            this->p7 = p7;
            this->p8 = p8;
            this->p9 = p9;
            this->p10 = p10;
            this->p11 = p11;
            this->p12 = p12;
            this->p13 = p13;
        }

        Key(const IdentifierType& id,
				    Parm1  &p1,  Parm2  &p2,  Parm3  &p3,  Parm4  &p4, Parm5  &p5,
				    Parm6  &p6,  Parm7  &p7,  Parm8  &p8,  Parm9  &p9, Parm10 &p10,
				    Parm11 &p11, Parm12 &p12, Parm13 &p13, Parm14 &p14) : count(14)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
            this->p5 = p5;
            this->p6 = p6;
            this->p7 = p7;
            this->p8 = p8;
            this->p9 = p9;
            this->p10 = p10;
            this->p11 = p11;
            this->p12 = p12;
            this->p13 = p13;
            this->p14 = p14;
        }

        Key(const IdentifierType& id,
				    Parm1  &p1,  Parm2  &p2,  Parm3  &p3,  Parm4  &p4,  Parm5  &p5,
				    Parm6  &p6,  Parm7  &p7,  Parm8  &p8,  Parm9  &p9,  Parm10 &p10,
				    Parm11 &p11, Parm12 &p12, Parm13 &p13, Parm14 &p14, Parm15 &p15) : count(15)
        {
            this->id = id;
            this->p1 = p1;
            this->p2 = p2;
            this->p3 = p3;
            this->p4 = p4;
            this->p5 = p5;
            this->p6 = p6;
            this->p7 = p7;
            this->p8 = p8;
            this->p9 = p9;
            this->p10 = p10;
            this->p11 = p11;
            this->p12 = p12;
            this->p13 = p13;
            this->p14 = p14;
            this->p15 = p15;
        }

        template<class F, typename I>
        friend bool operator==(const Key<F, I> &k1, const Key<F, I> &k2);
        
        template<class F, typename I>
        friend bool operator<(const Key<F, I> &k1, const Key<F, I> &k2);
    };

        
    template<class F, typename I>
     bool operator==(const Key<F, I> &k1, const Key<F, I> &k2)
    {
        if( k1.count != k2.count )
            return false;
        switch(k1.count){
            case -1:
                return true;
            case 0:
                if( k1.id == k2.id )
                    return true;
                else
                    return false;
            case 1:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) )
                    return true;
                else
                    return false;
            case 2:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) )
                    return true;
                else
                    return false;
            case 3:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) )
                    return true;
                else
                    return false;
            case 4:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) )
                    return true;
                else
                    return false;
            case 5:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) && 
                    (k1.p5 == k2.p5) )
                    return true;
                else
                    return false;
            case 6:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) && 
                    (k1.p5 == k2.p5) && 
                    (k1.p6 == k2.p6) )
                    return true;
                else
                    return false;
            case 7:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) && 
                    (k1.p5 == k2.p5) && 
                    (k1.p6 == k2.p6) && 
                    (k1.p7 == k2.p7) )
                    return true;
                else
                    return false;
            case 8:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) && 
                    (k1.p5 == k2.p5) && 
                    (k1.p6 == k2.p6) && 
                    (k1.p7 == k2.p7) && 
                    (k1.p8 == k2.p8) )
                    return true;
                else
                    return false;
            case 9:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) && 
                    (k1.p5 == k2.p5) && 
                    (k1.p6 == k2.p6) && 
                    (k1.p7 == k2.p7) && 
                    (k1.p8 == k2.p8) && 
                    (k1.p9 == k2.p9) )
                    return true;
                else
                    return false;
            case 10:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) && 
                    (k1.p5 == k2.p5) && 
                    (k1.p6 == k2.p6) && 
                    (k1.p7 == k2.p7) && 
                    (k1.p8 == k2.p8) && 
                    (k1.p9 == k2.p9) && 
                    (k1.p10 == k2.p10) )
                    return true;
                else
                    return false;
            case 11:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) && 
                    (k1.p5 == k2.p5) && 
                    (k1.p6 == k2.p6) && 
                    (k1.p7 == k2.p7) && 
                    (k1.p8 == k2.p8) && 
                    (k1.p9 == k2.p9) && 
                    (k1.p10 == k2.p10) && 
                    (k1.p11 == k2.p11) )
                    return true;
                else
                    return false;
            case 12:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) && 
                    (k1.p5 == k2.p5) && 
                    (k1.p6 == k2.p6) && 
                    (k1.p7 == k2.p7) && 
                    (k1.p8 == k2.p8) && 
                    (k1.p9 == k2.p9) && 
                    (k1.p10 == k2.p10) && 
                    (k1.p11 == k2.p11) && 
                    (k1.p12 == k2.p12) )
                    return true;
                else
                    return false;
            case 13:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) && 
                    (k1.p5 == k2.p5) && 
                    (k1.p6 == k2.p6) && 
                    (k1.p7 == k2.p7) && 
                    (k1.p8 == k2.p8) && 
                    (k1.p9 == k2.p9) && 
                    (k1.p10 == k2.p10) && 
                    (k1.p11 == k2.p11) && 
                    (k1.p12 == k2.p12) && 
                    (k1.p13 == k2.p13) )
                    return true;
                else
                    return false;
            case 14:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) && 
                    (k1.p5 == k2.p5) && 
                    (k1.p6 == k2.p6) && 
                    (k1.p7 == k2.p7) && 
                    (k1.p8 == k2.p8) && 
                    (k1.p9 == k2.p9) && 
                    (k1.p10 == k2.p10) && 
                    (k1.p11 == k2.p11) && 
                    (k1.p12 == k2.p12) && 
                    (k1.p13 == k2.p13) && 
                    (k1.p14 == k2.p14) )
                    return true;
                else
                    return false;
            case 15:
                if( (k1.id == k2.id) && 
                    (k1.p1 == k2.p1) && 
                    (k1.p2 == k2.p2) && 
                    (k1.p3 == k2.p3) && 
                    (k1.p4 == k2.p4) && 
                    (k1.p5 == k2.p5) && 
                    (k1.p6 == k2.p6) && 
                    (k1.p7 == k2.p7) && 
                    (k1.p8 == k2.p8) && 
                    (k1.p9 == k2.p9) && 
                    (k1.p10 == k2.p10) && 
                    (k1.p11 == k2.p11) && 
                    (k1.p12 == k2.p12) && 
                    (k1.p13 == k2.p13) && 
                    (k1.p14 == k2.p14) && 
                    (k1.p15 == k2.p15) )
                    return true;
                else
                    return false;
            default:
                return false;
        }
    }



    template<class F, typename I>
    bool operator<(const Key<F, I> &k1, const Key<F, I> &k2)
    {
        if( k1.count < k2.count )
            return true;
        switch(k1.count){
            case -1:
                return false;
            case 0:
                if( k1.id < k2.id )
                    return true;
                else
                    return false;
            case 1:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) )
                    return true;
                else
                    return false;
            case 2:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) )
                    return true;
                else
                    return false;
            case 3:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) )
                    return true;
                else
                    return false;
            case 4:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) )
                    return true;
                else
                    return false;
            case 5:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) || 
                    (k1.p5 < k2.p5) )
                    return true;
                else
                    return false;
            case 6:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) || 
                    (k1.p5 < k2.p5) || 
                    (k1.p6 < k2.p6) )
                    return true;
                else
                    return false;
            case 7:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) || 
                    (k1.p5 < k2.p5) || 
                    (k1.p6 < k2.p6) || 
                    (k1.p7 < k2.p7) )
                    return true;
                else
                    return false;
            case 8:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) || 
                    (k1.p5 < k2.p5) || 
                    (k1.p6 < k2.p6) || 
                    (k1.p7 < k2.p7) || 
                    (k1.p8 < k2.p8) )
                    return true;
                else
                    return false;
            case 9:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) || 
                    (k1.p5 < k2.p5) || 
                    (k1.p6 < k2.p6) || 
                    (k1.p7 < k2.p7) || 
                    (k1.p8 < k2.p8) || 
                    (k1.p9 < k2.p9) )
                    return true;
                else
                    return false;
            case 10:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) || 
                    (k1.p5 < k2.p5) || 
                    (k1.p6 < k2.p6) || 
                    (k1.p7 < k2.p7) || 
                    (k1.p8 < k2.p8) || 
                    (k1.p9 < k2.p9) || 
                    (k1.p10 < k2.p10) )
                    return true;
                else
                    return false;
            case 11:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) || 
                    (k1.p5 < k2.p5) || 
                    (k1.p6 < k2.p6) || 
                    (k1.p7 < k2.p7) || 
                    (k1.p8 < k2.p8) || 
                    (k1.p9 < k2.p9) || 
                    (k1.p10 < k2.p10) || 
                    (k1.p11 < k2.p11) )
                    return true;
                else
                    return false;
            case 12:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) || 
                    (k1.p5 < k2.p5) || 
                    (k1.p6 < k2.p6) || 
                    (k1.p7 < k2.p7) || 
                    (k1.p8 < k2.p8) || 
                    (k1.p9 < k2.p9) || 
                    (k1.p10 < k2.p10) || 
                    (k1.p11 < k2.p11) || 
                    (k1.p12 < k2.p12) )
                    return true;
                else
                    return false;
            case 13:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) || 
                    (k1.p5 < k2.p5) || 
                    (k1.p6 < k2.p6) || 
                    (k1.p7 < k2.p7) || 
                    (k1.p8 < k2.p8) || 
                    (k1.p9 < k2.p9) || 
                    (k1.p10 < k2.p10) || 
                    (k1.p11 < k2.p11) || 
                    (k1.p12 < k2.p12) || 
                    (k1.p13 < k2.p13) )
                    return true;
                else
                    return false;
            case 14:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) || 
                    (k1.p5 < k2.p5) || 
                    (k1.p6 < k2.p6) || 
                    (k1.p7 < k2.p7) || 
                    (k1.p8 < k2.p8) || 
                    (k1.p9 < k2.p9) || 
                    (k1.p10 < k2.p10) || 
                    (k1.p11 < k2.p11) || 
                    (k1.p12 < k2.p12) || 
                    (k1.p13 < k2.p13) || 
                    (k1.p14 < k2.p14) )
                    return true;
                else
                    return false;
            case 15:
                if( (k1.id < k2.id) || 
                    (k1.p1 < k2.p1) || 
                    (k1.p2 < k2.p2) || 
                    (k1.p3 < k2.p3) || 
                    (k1.p4 < k2.p4) || 
                    (k1.p5 < k2.p5) || 
                    (k1.p6 < k2.p6) || 
                    (k1.p7 < k2.p7) || 
                    (k1.p8 < k2.p8) || 
                    (k1.p9 < k2.p9) || 
                    (k1.p10 < k2.p10) || 
                    (k1.p11 < k2.p11) || 
                    (k1.p12 < k2.p12) || 
                    (k1.p13 < k2.p13) || 
                    (k1.p14 < k2.p14) || 
                    (k1.p15 < k2.p15) )
                    return true;
                else
                    return false;
            default:
                return false;
        }
    }

    

} // namespace Loki

#endif // end file guardian

