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

#ifndef TEUCHOS_VERBOSE_OBJECT_HPP
#define TEUCHOS_VERBOSE_OBJECT_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_VerbosityLevel.hpp"


namespace Teuchos {


/** \brief Non-templated base class for objects that can print their
 * activities to a stream.
 *
 * \ingroup teuchos_outputting_grp
 *
 * Objects that derive from this interface print to a default class-owned
 * (i.e. static) output stream object (set using <tt>setDefaultOStream()</tt>)
 * or the output stream can be set on an object-by-object basis using
 * <tt>setOStream()</tt>.
 *
 * The output stream type is <tt>FancyOStream</tt> which allows for automated
 * indentation (using the <tt>OSTab</tt> class) and has other useful features.
 */
class VerboseObjectBase {
public:

  //! @name Public static member functions 
  //@{

  /** \brief Set the default output stream object.
   *
   * If this function is not called, then a default stream based on
   * <tt>std::cout</tt> is used.
   */
  static void setDefaultOStream( const RCP<FancyOStream> &defaultOStream );

  /** \brief Get the default output stream object. */
  static RCP<FancyOStream> getDefaultOStream();

  //@}

  //! @name Constructors/Initializers 
  //@{
  
  /** \brief . */
  virtual ~VerboseObjectBase() {}
  
  /** \brief Calls <tt>initializeVerboseObject()</tt>.
   */
  explicit
  VerboseObjectBase(
    const RCP<FancyOStream> &oStream = Teuchos::null
    );
  
  /** \brief Calls <tt>initializeVerboseObject()</tt>.
   */
  virtual void initializeVerboseObjectBase(
    const RCP<FancyOStream> &oStream = Teuchos::null
    );

  /** \brief The output stream for <tt>*this</tt> object.
   *
   * This function is supposed by called by general clients to set the output
   * stream according to some general logic in the code.
   */
  virtual const VerboseObjectBase& setOStream(
    const RCP<FancyOStream> &oStream) const;

  /** \brief Set the overriding the output stream for <tt>*this</tt> object.
   *
   * This function is supposed to be called by special clients that want to
   * set the output stream in a way that will not be overridden by
   * <tt>setOStream()</tt>.
   */
  virtual const VerboseObjectBase& setOverridingOStream(
    const RCP<FancyOStream> &oStream) const;

  /** \brief Set line prefix name for this object */
  virtual VerboseObjectBase& setLinePrefix(const std::string &linePrefix);

  //@}

  //! @name Query functions 
  //@{

  /** \brief Return the output stream to be used for out for <tt>*this</tt>
   * object.
   */
  virtual RCP<FancyOStream> getOStream() const;

  /** \brief Return the the overriding output stream if set.
   *
   * This is the output stream tthat will be returned from
   * <tt>getOStream()</tt> regardless that stream is set by
   * <tt>setOStream()</tt>.
   */
  virtual RCP<FancyOStream> getOverridingOStream() const;

  /** \brief Get the line prefix for this object */
  virtual std::string getLinePrefix() const;

  //@}

  //! @name Utilities 
  //@{

  /** \brief Create a tab object which sets the number of tabs and optionally the line prefix.
   *
   * \param tabs  [in] The number of relative tabs to add (if <tt>tabs > 0</tt>) or remove (if <tt>tabs < 0</tt>).
   *              If <tt>tabs == OSTab::DISABLE_TABBING</tt> then tabbing will be turned off temporarily.
   *
   * \param linePrefix
   *              [in] Sets a line prefix that overrides <tt>this->getLinePrefix()</tt>.
   *
   * The side effects of these changes go away as soon as the returned
   * <tt>OSTab</tt> object is destroyed at the end of the block of code.
   *
   * Returns <tt>OSTab( this->getOStream(), tabs, linePrefix.length() ? linePrefix : this->getLinePrefix() )</tt>
   */
  virtual OSTab getOSTab(const int tabs = 1, const std::string &linePrefix = "") const;

  //@}

protected:
  
  /** \brief Function that is called whenever the verbosity state
   * is updated.
   *
   * Subclasses can override this function to be informed whenever the
   * verbosity state of <tt>*this</tt> object gets updated.
   *
   * The default implementation simply does nothing.
   */
  virtual void informUpdatedVerbosityState() const;

private:

  mutable RCP<FancyOStream> thisOStream_;
  mutable RCP<FancyOStream> thisOverridingOStream_;
  std::string thisLinePrefix_;

  static RCP<FancyOStream>& privateDefaultOStream();

};


/** \brief Templated base class for objects that can print their activities to
 * a stream and have a verbosity level.
 *
 * Objects that derive from this interface print to a default class-owned
 * (i.e. static) output stream object (set using <tt>setDefaultOStream()</tt>)
 * or the output stream can be set on an object-by-object basis using
 * <tt>setOStream()</tt> .  In addition, each object, by default, has a
 * verbosity level that is shared by all objects (set using
 * <tt>setDefaultVerbosityLevel()</tt>) or can be set on an object-by-object
 * basis using <tt>setVerbLevel()</tt>.
 *
 * The output stream type is <tt>FancyOStream</tt> which allows for automated
 * indentation (using the <tt>OSTab</tt> class) and has other useful features.
 *
 * Note that <tt>setOStream()</tt> and <tt>setVerbLevel()</tt> are actually
 * decalared as <tt>const</tt> functions.  This is to allow a client to
 * temporarily change the stream and verbosity level.  To do this saftely, use
 * the class <tt>VerboseObjectTempState</tt> which will revert the output
 * state after it is destroyed.
 */
template<class ObjectType>
class VerboseObject : virtual public VerboseObjectBase {
public:

  //! @name Public static member functions 
  //@{

  /** \brief Set the default verbosity level.
   *
   * If not called, then the default verbosity level is <tt>VERB_DEFAULT</tt>
   */
  static void setDefaultVerbLevel( const EVerbosityLevel defaultVerbLevel);

  /** \brief Get the default verbosity level. */
  static EVerbosityLevel getDefaultVerbLevel();

  //@}

  //! @name Constructors/Initializers 
  //@{
  
  /** \brief Calls <tt>initializeVerboseObject()</tt>.
   */
  explicit
  VerboseObject(
    const EVerbosityLevel verbLevel = VERB_DEFAULT,  // Note, this must be the same as the default value for defaultVerbLevel_
    const RCP<FancyOStream> &oStream  = Teuchos::null
    );
  
  /** \brief Calls <tt>initializeVerboseObject()</tt>.
   */
  virtual void initializeVerboseObject(
    const EVerbosityLevel verbLevel = VERB_DEFAULT,  // Note, this must be the same as the default value for defaultVerbLevel_
    const RCP<FancyOStream> &oStream  = Teuchos::null
    );

  /** \brief Set the verbosity level for <tt>*this</tt> object.
   *
   *
   * This function is supposed by called by general clients to set the output
   * level according to some general logic in the code.
   */
  virtual const VerboseObject& setVerbLevel(
    const EVerbosityLevel verbLevel) const;

  /** \brief Set the overriding verbosity level for <tt>*this</tt> object.
   *
   * This function is supposed to be called by special clients that want to
   * set the output level in a way that will not be overridden by
   * <tt>setOStream()</tt>.
   */
  virtual const VerboseObject& setOverridingVerbLevel(
    const EVerbosityLevel verbLevel) const;

  //@}

  //! @name Query functions 
  //@{

  /** \brief Get the verbosity level */
  virtual EVerbosityLevel getVerbLevel() const;

  //@}
  
private:
  
  mutable EVerbosityLevel thisVerbLevel_;
  mutable EVerbosityLevel thisOverridingVerbLevel_;
  
  static EVerbosityLevel& privateDefaultVerbLevel();

};


/** \brief Set and release a stream and verbosity level.
 *
 */
template<class ObjectType>
class VerboseObjectTempState {
public:
  /** \brief . */
  VerboseObjectTempState(
    const RCP<const VerboseObject<ObjectType> > &verboseObject,
    const RCP<FancyOStream> &newOStream,
    const EVerbosityLevel newVerbLevel
    )
    :verboseObject_(verboseObject)
    {
      if(verboseObject_.get()) {
        oldOStream_ = verboseObject_->getOStream();
        oldVerbLevel_ = verboseObject_->getVerbLevel();
        verboseObject_->setOStream(newOStream);
        verboseObject_->setVerbLevel(newVerbLevel);
      }
    }
  /** \brief . */
  ~VerboseObjectTempState()
    {
      if(verboseObject_.get()) {
        verboseObject_->setOStream(oldOStream_);
        verboseObject_->setVerbLevel(oldVerbLevel_);
      }
    }
private:
  RCP<const VerboseObject<ObjectType> > verboseObject_;
  RCP<FancyOStream> oldOStream_;
  EVerbosityLevel oldVerbLevel_;
  // Not defined and not to be called
  VerboseObjectTempState();
  VerboseObjectTempState(const VerboseObjectTempState&);
  VerboseObjectTempState& operator=(const VerboseObjectTempState&);
};


// //////////////////////////////////
// Template defintions


//
// VerboseObject
//


// Public static member functions


template<class ObjectType>
void VerboseObject<ObjectType>::setDefaultVerbLevel( const EVerbosityLevel defaultVerbLevel)
{
  privateDefaultVerbLevel() = defaultVerbLevel;
}


template<class ObjectType>
EVerbosityLevel VerboseObject<ObjectType>::getDefaultVerbLevel()
{
  return privateDefaultVerbLevel();
}


// Constructors/Initializers


template<class ObjectType>
VerboseObject<ObjectType>::VerboseObject(
  const EVerbosityLevel verbLevel,
  const RCP<FancyOStream> &oStream
  )
  : thisOverridingVerbLevel_(VERB_DEFAULT)
{
  this->initializeVerboseObject(verbLevel,oStream);
}


template<class ObjectType>
void VerboseObject<ObjectType>::initializeVerboseObject(
  const EVerbosityLevel verbLevel,
  const RCP<FancyOStream> &oStream
  )
{
  thisVerbLevel_ = verbLevel;
  this->initializeVerboseObjectBase(oStream);
}


template<class ObjectType>
const VerboseObject<ObjectType>&
VerboseObject<ObjectType>::setVerbLevel(const EVerbosityLevel verbLevel) const
{
  thisVerbLevel_ = verbLevel;
  informUpdatedVerbosityState();
  return *this;
}


template<class ObjectType>
const VerboseObject<ObjectType>&
VerboseObject<ObjectType>::setOverridingVerbLevel(
  const EVerbosityLevel verbLevel
  ) const
{
  thisOverridingVerbLevel_ = verbLevel;
  informUpdatedVerbosityState();
  return *this;
}


// Query functions


template<class ObjectType>
EVerbosityLevel VerboseObject<ObjectType>::getVerbLevel() const
{
  if (VERB_DEFAULT != thisOverridingVerbLevel_)
    return thisOverridingVerbLevel_;
  if (VERB_DEFAULT == thisVerbLevel_)
    return getDefaultVerbLevel();
  return thisVerbLevel_;
}


// Private static members


template<class ObjectType>
EVerbosityLevel& VerboseObject<ObjectType>::privateDefaultVerbLevel()
{
  static EVerbosityLevel defaultVerbLevel = VERB_DEFAULT;
  return defaultVerbLevel;
}


} // namespace Teuchos


#endif // TEUCHOS_VERBOSE_OBJECT_HPP
