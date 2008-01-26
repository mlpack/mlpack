//@HEADER
/*
************************************************************************

              Epetra: Linear Algebra Services Package 
                Copyright (2001) Sandia Corporation

Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
license for use of this work by or on behalf of the U.S. Government.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2.1 of the
License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA
Questions? Contact Michael A. Heroux (maherou@sandia.gov) 

************************************************************************
*/
//@HEADER

#ifndef Epetra_HashTable_H_
#define Epetra_HashTable_H_

#include "Epetra_Object.h"

class Epetra_HashTable : public Epetra_Object
{
  struct Node
  {
     int Key;
     int Value;
     Node * Ptr;

     Node( const int key = 0, const int value = 0, Node * ptr = 0 )
     : Key(key), Value(value), Ptr(ptr) {}

    private:
     Node(const Node& src)
       : Key(src.Key), Value(src.Value), Ptr(src.Ptr) {}

    Node& operator=(const Node& src)
    { Key = src.Key; Value = src.Value; Ptr = src.Ptr; return(*this); }
  };

  Node ** Container_;
  int Size_;
  unsigned int Seed_;

  int Func( const int key ) { return (Seed_ ^ key)%Size_; }
     
 public:

  Epetra_HashTable( const int size, const unsigned int seed = (2654435761U) )
  : Container_(NULL),
    Size_(size),
    Seed_(seed)
  {
    if (size<=0)
      throw ReportError( "Bad Hash Table Size: " + toString(size), -1 );

    Container_ = new Node * [size];
    for( int i = 0; i < size; ++i ) Container_[i] = 0;
  }

  Epetra_HashTable( const Epetra_HashTable & obj )
  : Container_(NULL),
    Size_(obj.Size_),
    Seed_(obj.Seed_)
  {
    Container_ = new Node * [Size_];
    for( int i = 0; i < Size_; ++i ) Container_[i] = 0;
    for( int i = 0; i < Size_; ++i )
    {
      Node * ptr = obj.Container_[i];
      while( ptr ) { Add( ptr->Key, ptr->Value ); ptr = ptr->Ptr; }
    }
  }

  ~Epetra_HashTable()
  {
    Node * ptr1;
    Node * ptr2;
    for( int i = 0; i < Size_; ++i )
    {
      ptr1 = Container_[i];
      while( ptr1 ) { ptr2 = ptr1; ptr1 = ptr1->Ptr; delete ptr2; }
    }

    delete [] Container_;
  }

  void Add( const int key, const int value )
  {
    int v = Func(key);
    Node * n1 = Container_[v];
    Container_[v] = new Node(key,value,n1);
  }

  int Get( const int key )
  {
    Node * n = Container_[ Func(key) ];
    while( n && (n->Key != key) ) n = n->Ptr;
    if( n ) return n->Value;
    else    return -1;
  }

 private:
  Epetra_HashTable& operator=(const Epetra_HashTable& src)
    {
      (void)src;
      //not currently supported
      bool throw_error = true;
      if (throw_error) {
	throw ReportError("Epetra_HashTable::operator= not supported.",-1);
      }
      return(*this);
    }

};

#endif
