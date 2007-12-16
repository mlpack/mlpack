// @HEADER
// ***********************************************************************
// 
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ***********************************************************************
// @HEADER

#ifndef TEUCHOS_SERIALIZATION_TRAITS_HPP
#define TEUCHOS_SERIALIZATION_TRAITS_HPP

#include "Teuchos_ConfigDefs.hpp"

namespace Teuchos {

template<typename T>
struct UndefinedSerializationTraits {
  //! This function should not compile if there is an attempt to instantiate!
  static inline T notDefined() {return(T::this_type_is_missing_a_specialization());}
};

/** \brief Serialization traits class for types that use value semantics.
 *
 * There are one of two modes associated with serialization.
 * 
 * ToDo: Finish documenation!
 */
template <typename Ordinal, typename T>
class SerializationTraits {
public:
  
  //! @name Seialization type selection 
  //@{

  /** \brief Determines if the type supports direct serialization. */
  static const bool supportsDirectSerialization = false;

  //@}

  //! @name Direct serialization functions (not defined if supportsDirectSerialization==false) 
  //@{

  /** \brief Return the number of bytes for <tt>count</tt> objects. */
  static Ordinal fromCountToDirectBytes(const Ordinal count) { UndefinedSerializationTraits<T>::notDefined(); return 0; }

  /** \brief Convert the pointer type to <tt>char*</tt>. */
  static char* convertToCharPtr( T* ptr ) { UndefinedSerializationTraits<T>::notDefined(); return 0; }

  /** \brief Convert the pointer type to <tt>const char*</tt>. */
  static const char* convertToCharPtr( const T* ptr ) { UndefinedSerializationTraits<T>::notDefined(); return 0; }

  /** \brief Return the number of objects for <tt>bytes</tt> of storage. */
  static Ordinal fromDirectBytesToCount(const Ordinal bytes) { UndefinedSerializationTraits<T>::notDefined(); return 0; }

  /** \brief Convert the pointer type from <tt>char*</tt>. */
  static T* convertFromCharPtr( char* ptr ) { UndefinedSerializationTraits<T>::notDefined(); return 0; }

  /** \brief Convert the pointer type from <tt>char*</tt>. */
  static const T* convertFromCharPtr( const char* ptr ) { UndefinedSerializationTraits<T>::notDefined(); return 0; }

  //@}

  //! @name Indirect serialization functions (always defined and supported) 
  //@{

  /** \brief Return the number of bytes for <tt>count</tt> objects. */
  static Ordinal fromCountToIndirectBytes(const Ordinal count) { UndefinedSerializationTraits<T>::notDefined(); return 0; }

  /** \brief Serialize to an indirect <tt>char[]</tt> buffer.
   *
   * \param  count
   *           [in] The number of objects to serialize.
   * \param  buffer
   *           [in] The objects to serialize.
   * \param  bytes
   *           [in] Number of bytes in <tt>charBuffer[]</tt>
   * \param  charBuffer
   *           [out] Array (length <tt>bytes</tt>) containing the serialized objects.
   *
   * <b>Preconditions:</b><ul>
   * <li><tt>bytes==fromCountToIndirectBytes(count)</tt>
   * </ul>
   */
  static void serialize(
    const Ordinal count, const T buffer[], const Ordinal bytes, char charBuffer[]
    )
    { UndefinedSerializationTraits<T>::notDefined(); }

  /** \brief Return the number of objects for <tt>bytes</tt> of storage. */
  static Ordinal fromIndirectBytesToCount(const Ordinal bytes) { UndefinedSerializationTraits<T>::notDefined(); return 0; }

  /** \brief Deserialize from an indirect <tt>char[]</tt> buffer.
   *
   * \param  bytes
   *           [in] Number of bytes in <tt>charBuffer[]</tt>
   * \param  charBuffer
   *           [in] Array (length <tt>bytes</tt>) containing the serialized objects.
   * \param  count
   *           [in] The number of objects to deserialize.
   * \param  buffer
   *           [out] The deserialized objects.
   
   * <b>Preconditions:</b><ul>
   * <li><tt>count==fromIndirectBytesToCount(bytes)</tt>
   * </ul>
   */
  static void deserialize(
    const Ordinal bytes, const char charBuffer[], const Ordinal count, T buffer[]
    )
    { UndefinedSerializationTraits<T>::notDefined(); }
  
  //@}

};

template <typename Ordinal, typename T>
class DirectSerializationTraits {
public:
  static const bool supportsDirectSerialization = true;
  // Direct serialization
  static Ordinal fromCountToDirectBytes(const Ordinal count)
    { return sizeof(T)*count; }
  static char* convertToCharPtr( T* ptr )
    { return reinterpret_cast<char*>(ptr); }
  static const char* convertToCharPtr( const T* ptr )
    { return reinterpret_cast<const char*>(ptr); }
  static Ordinal fromDirectBytesToCount(const Ordinal count)
    { return count/sizeof(T); }
  static T* convertFromCharPtr( char* ptr )
    { return reinterpret_cast<T*>(ptr); }
  static const T* convertFromCharPtr( const char* ptr )
    { return reinterpret_cast<const T*>(ptr); }
  // Indirect serialization
  static Ordinal fromCountToIndirectBytes(const Ordinal count)
    { return fromCountToDirectBytes(count); }
  static void serialize(
    const Ordinal count, const T buffer[], const Ordinal bytes, char charBuffer[]
    )
    {
#ifdef TEUCHOS_DEBUG
      TEST_FOR_EXCEPT(bytes!=fromCountToIndirectBytes(count));
#endif
      const char *_buffer = convertToCharPtr(buffer);
      std::copy(_buffer,_buffer+bytes,charBuffer);
    }
  static Ordinal fromIndirectBytesToCount(const Ordinal bytes) 
    { return fromDirectBytesToCount(bytes); }
  static void deserialize(
    const Ordinal bytes, const char charBuffer[], const Ordinal count, T buffer[]
    )
    {
#ifdef TEUCHOS_DEBUG
      TEST_FOR_EXCEPT(count!=fromIndirectBytesToCount(bytes));
#endif
      char *_buffer = convertToCharPtr(buffer);
      std::copy(charBuffer,charBuffer+bytes,_buffer);
    }
};

template<typename Ordinal>
class SerializationTraits<Ordinal,char>
  : public DirectSerializationTraits<Ordinal,char>
{};

template<typename Ordinal>
class SerializationTraits<Ordinal,int>
  : public DirectSerializationTraits<Ordinal,int>
{};

template<typename Ordinal>
class SerializationTraits<Ordinal,float>
  : public DirectSerializationTraits<Ordinal,float>
{};

template<typename Ordinal>
class SerializationTraits<Ordinal,double>
  : public DirectSerializationTraits<Ordinal,double>
{};

#if defined(HAVE_COMPLEX) && defined(HAVE_TEUCHOS_COMPLEX)

template<typename Ordinal>
class SerializationTraits<Ordinal,std::complex<float> >
  : public DirectSerializationTraits<Ordinal,std::complex<float> >
{};

template<typename Ordinal>
class SerializationTraits<Ordinal,std::complex<double> >
  : public DirectSerializationTraits<Ordinal,std::complex<double> >
{};

#endif // defined(HAVE_COMPLEX) && defined(HAVE_TEUCHOS_COMPLEX)

} // namespace Teuchos

#endif // TEUCHOS_SERIALIZATION_TRAITS_HPP
