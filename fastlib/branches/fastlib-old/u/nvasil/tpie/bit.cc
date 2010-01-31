// Copyright (c) 1994 Darren Vengroff
//
// File: bit.cpp
// Author: Darren Vengroff <darrenv@eecs.umich.edu>
// Created: 11/4/94
//

#include <versions.h>
VERSION(bit_cpp,"$Id: bit.cpp,v 1.5 2003/09/12 18:46:44 jan Exp $");

#include <bit.h>

bit::bit(void)
{
}

bit::bit(bool b)
{
    data = (b == true);
}

bit::bit(int i)
{
    data = (i != 0);
}

bit::bit(long int i)
{
    data = (i != 0);
}

bit::operator bool(void)
{
    return (data != 0);
}
        
bit::operator int(void)
{
    return data;
}
        
bit::operator long int(void)
{
    return data;
}
        
bit::~bit(void)
{
}

bit bit::operator+=(bit rhs)
{
    return *this = *this + rhs;
}
        
bit bit::operator*=(bit rhs)
{
    return *this = *this + rhs;
}

bit operator+(bit op1, bit op2)
{
    return bit(op1.data ^ op2.data);
}


bit operator*(bit op1, bit op2)
{
    return bit(op1.data & op2.data);
}


ostream &operator<<(ostream &s, bit b)
{
    return s << int(b.data);
}




