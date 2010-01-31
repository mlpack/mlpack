#ifndef TEUCHOS_ANY_HPP
#define TEUCHOS_ANY_HPP

/*! \file Teuchos_any.hpp
   \brief Modified boost::any class for holding a templated value
*/

#include "Teuchos_TestForException.hpp"
#include "Teuchos_TypeNameTraits.hpp"

//
// This file was taken from the boost library which contained the
// following notice:
//
// *************************************************************
//
// what:  variant type boost::any
// who:   contributed by Kevlin Henney,
//        with features contributed and bugs found by
//        Ed Brey, Mark Rodgers, Peter Dimov, and James Curran
// when:  July 2001
// where: tested with BCC 5.5, MSVC 6.0, and g++ 2.95
//
// Copyright Kevlin Henney, 2000, 2001, 2002. All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.
//
// *************************************************************
//
// RAB modified the file for use in Teuchos.  I changed the nature of
// the any_cast<> to be easier to use.
//

namespace Teuchos {

/** \brief Modified boost::any class, which is a container for a templated
 * value.
 */
class any
{
public:
  //! Empty constructor
  any()
    : content(0)
    {}

  //! Templated constructor
  template<typename ValueType>
  explicit any(const ValueType & value)
    : content(new holder<ValueType>(value))
    {}
  
  //! Copy constructor
  any(const any & other)
    : content(other.content ? other.content->clone() : 0)
    {}

  //! Destructor
  ~any()
    {
      delete content;
    }

  //! Method for swapping the contents of two any classes
  any & swap(any & rhs)
    {
      std::swap(content, rhs.content);
      return *this;
    }
  
  //! Copy the value <tt>rhs</tt>
  template<typename ValueType>
  any & operator=(const ValueType & rhs)
    {
      any(rhs).swap(*this);
      return *this;
    }
  
  //! Copy the value held in <tt>rhs</tt>
  any & operator=(const any & rhs)
    {
      any(rhs).swap(*this);
      return *this;
    }
  
  //! Return true if nothing is being stored
  bool empty() const
    {
      return !content;
    }
  
  //! Return the type of value being stored
  const std::type_info & type() const
    {
      return content ? content->type() : typeid(void);
    }
  
  //! Return the name of the type
  std::string typeName() const
    {
      return content ? content->typeName() : "NONE";
    }
  
  //! \brief Return if two any objects are the same or not. 
  bool same( const any &other ) const
    {
      if( this->empty() && other.empty() )
        return true;
      else if( this->empty() && !other.empty() )
        return false;
      else if( !this->empty() && other.empty() )
        return false;
      // !this->empty() && !other.empty()
      return content->same(*other.content);
    }

  //! Print this value to the output stream <tt>os</tt>
  void print(std::ostream& os) const
    {
      if (content) content->print(os);
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  /** @name Private??? types */
  //@{

  /** \brief . */
  class placeholder
  {
  public:
    /** \brief . */
    virtual ~placeholder() {}
    /** \brief . */
    virtual const std::type_info & type() const = 0;
    /** \brief . */
    virtual std::string typeName() const = 0;
    /** \brief . */
    virtual placeholder * clone() const = 0;
    /** \brief . */
    virtual bool same( const placeholder &other ) const = 0;
    /** \brief . */
    virtual void print(std::ostream & os) const = 0;
  };
  
  /** \brief . */
  template<typename ValueType>
  class holder : public placeholder
  {
  public:
    /** \brief . */
    holder(const ValueType & value)
      : held(value)
      {}
    /** \brief . */
    const std::type_info & type() const
      { return typeid(ValueType); }
    /** \brief . */
    std::string typeName() const
      { return TypeNameTraits<ValueType>::name(); }
    /** \brief . */
    placeholder * clone() const
      { return new holder(held); }
    /** \brief . */
    bool same( const placeholder &other ) const
      {
        if( type() != other.type() ) {
          return false;
        }
        // type() == other.type()
        const ValueType
          &other_held = dynamic_cast<const holder<ValueType>&>(other).held;
        return held == other_held;
      }
    /** \brief . */
    void print(std::ostream & os) const
      { os << held; }
    /** \brief . */
    ValueType held;
  };

  //@}

public:
  // Danger: This is made public to allow any_cast to be non-friend
  placeholder* access_content()
    { return content; }
  const placeholder* access_content() const
    { return content; }
#endif

private:

  // /////////////////////////
  // Private data members
  
  placeholder * content;

};

/*! \relates any
    \brief Thrown if any_cast is attempted between two incompatable types.
*/
class bad_any_cast : public std::runtime_error
{
public:
  bad_any_cast( const std::string msg ) : std::runtime_error(msg) {}
};

/*! \relates any
    \brief Used to extract the templated value held in Teuchos::any to a given value type.

    \note <ul>   <li> If the templated value type and templated type are not the same then a 
    bad_any_cast is thrown.
    <li> If the dynamic cast fails, then a Teuchos::bad_any_cast std::exception is thrown.
    </ul>
*/
template<typename ValueType>
ValueType& any_cast(any &operand)
{

  const std::string ValueTypeName = TypeNameTraits<ValueType>::name();
  bool type_mismatch = ( operand.type() != typeid(ValueType) );
#ifdef HAVE_SHARED
  // For shared libraries, the above may resolve as true because one
  // of the types is not fully qualified.  The following check will
  // compensate for that.
  if ( type_mismatch )
    type_mismatch = ( strcmp(operand.type().name(), typeid(ValueType).name()) != 0 );
#endif
  TEST_FOR_EXCEPTION(
    type_mismatch, bad_any_cast
    ,"any_cast<"<<ValueTypeName<<">(operand): Error, cast to type "
    << "any::holder<"<<ValueTypeName<<"> failed since the actual underlying type is \'"
    << typeName(*operand.access_content()) << "!"
    );

  TEST_FOR_EXCEPTION(
    !operand.access_content(), bad_any_cast
    ,"any_cast<"<<ValueTypeName<<">(operand): Error, cast to type "
    << "any::holder<"<<ValueTypeName<<"> failed because the content is NULL"
    );

  // 2007/11/28: rabartl: Note: If we get here, then the rest of this function
  // should succeed and should be a formality.  The dynamic cast is the safe
  // thing to do and therefore we try that.  However, this may actually fail
  // with shared libraries with g++ (see below).
  any::holder<ValueType>
    *dyn_cast_content = dynamic_cast<any::holder<ValueType>*>(operand.access_content());
#ifdef HAVE_SHARED
  // For shared libraries, dynamic_cast does not work across shared
  // library boundaries, so we use static_cast, which will prevent any
  // runtime errors at the expense of being more dangerous.
  if ( !dyn_cast_content )
    dyn_cast_content = static_cast<any::holder<ValueType>*>(operand.access_content());
  // 2007/11/28: rabartl: Above, this static cast is not safe for some
  // possible types but that should be very rare.  Note that if a program
  // performing the dynamic casts works correctly with a static library
  // (i.e. HAVE_SHARED is not defined), and if the compiler can figure out the
  // static case correctly, then this should be safe for the shared library
  // case.  We really need to look into issues of RTTI and shared libraries
  // and see what the C++ standard says about this and if there are any g++
  // command-line options for fixing this ...
#endif
  TEST_FOR_EXCEPTION(
    !dyn_cast_content, std::logic_error
    ,"any_cast<"<<ValueTypeName <<">(operand): Error, cast to type "
    << "any::holder<"<<ValueTypeName<<"> failed but should not have and the actual underlying type is \'"
    << typeName(*operand.access_content()) << "!"
    );

  return dyn_cast_content->held;

}

/*! \relates any
    \brief Used to extract the const templated value held in Teuchos::any to a given 
  const value type.

    \note <ul>   <li> If the templated value type and templated type are not the same then 
    bad_any_cast is thrown.
    <li> If the dynamic cast fails, then a logic_error is thrown.
    </ul>
*/
template<typename ValueType>
const ValueType& any_cast(const any &operand)
{
  return any_cast<ValueType>(const_cast<any&>(operand));
}

/*! \relates any
    \brief Converts the value in <tt>any</tt> to a std::string.
*/
inline std::string toString(const any &rhs)
{
  std::ostringstream oss;
  rhs.print(oss);
  return oss.str();
}

/*! \relates any
    \brief Returns true if two any objects have the same value.
*/
inline bool operator==( const any &a, const any &b )
{
  return a.same(b);
}

/*! \relates any
    \brief Returns true if two any objects <b>do not</tt> have the same value.
*/
inline bool operator!=( const any &a, const any &b )
{
  return !a.same(b);
}

/*! \relates any
    \brief Writes "any" input <tt>rhs</tt> to the output stream <tt>os</tt>.
*/
inline std::ostream & operator<<(std::ostream & os, const any &rhs)
{
  rhs.print(os);
  return os;
}

} // namespace Teuchos

#endif // TEUCHOS_ANY_HPP
