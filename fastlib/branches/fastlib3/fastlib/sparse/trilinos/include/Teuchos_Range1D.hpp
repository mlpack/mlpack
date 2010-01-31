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

// Range1D class used for representing a range of positive integers.
// Its primary usage is in accessing vectors and matrices by subregions
// of rows and columns
//

#ifndef TEUCHOS_RANGE1D_HPP
#define TEUCHOS_RANGE1D_HPP

/*! \file Teuchos_Range1D.hpp
    \brief .
*/

#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_TestForException.hpp"

namespace Teuchos {

/** \brief Subregion Index Range Class.
 * 
 * The class <tt>%Range1D</tt> abstracts a 1-D, zero-based, range of indexes.
 * It is used to index into vectors and matrices and return subregions of them
 * respectively.
 *
 * Constructing using <tt>Range1D()</tt> yields a range that represents the
 * entire dimension of an object <tt>[0, max_ubound-1]</tt> (an entire std::vector,
 * all the rows in a matrix, or all the columns in a matrix etc.).
 *
 * Constructing using <tt>\ref Range1D::Range1D "Range1D(INVALID)"</tt> yields
 * an invalid range <tt>[0,-1]</tt> with <tt>size() == 0</tt>.  In fact the
 * condition <tt>size()==0</tt> is the determining flag that a range is not
 * valid.  Once constructed with <tt>Range1D(INVALID)</tt>, a
 * <tt>%Range1D</tt> object can pass through many other operations that may
 * change <tt>%lbound()</tt> and <tt>%ubound()</tt> but will never change
 * <tt>size()==0</tt>.
 *
 * Constructing using <tt>\ref Range1D::Range1D "Range1D(lbound,ubound)"</tt>
 * yields a finite-dimensional zero-based range.  The validity of constructed
 * range will only be checked if <tt>TEUCHOS_DEBUG</tt> is defined.
 *
 * There are many \ref Range1D_funcs_grp "non-member functions" that can be
 * used with <tt>%Range1D</tt> objects.
 *
 * The default copy constructor and assignment operator functions are allowed
 * since they have the correct semantics.
 */
class Range1D {
public:

  /** \brief . */
  typedef Teuchos_Index  Index;

  /** \brief . */
  enum EInvalidRange { INVALID };

  /** \brief Used for Range1D(INVALID) */
  static const Range1D Invalid;

  /** \brief Constructs a range representing the entire range.
   *
   * Postconditions: <ul>
   *	<li> <tt>this->full_range()==true</tt>
   * <li> <tt>this->size()</tt> is a very large number
   * <li> <tt>this->lbound()==0</tt>
   * <li> <tt>this->ubound()</tt> is a very large number
   *	</ul>
   */
  inline Range1D();

  /** Constructs an invalid (zero) range.
   *
   * Postconditions: <ul>
   *	<li> <tt>this->full_range() == false</tt>
   * <li> <tt>this->size() == 0</tt>
   * <li> <tt>this->lbound()==0</tt>
   * <li> <tt>this->ubound()==-1</tt>
   *	</ul>
   */
  inline Range1D( EInvalidRange );

  /** \brief Constructs a range that represents the range <tt>[lbound, ubound]</tt>.
   *
   * Preconditions: <ul>
   *	<li> <tt>lbound >= 0</tt> (throw \c range_error)
   *	<li> <tt>lbound <= ubound</tt> (throw \c range_error)
   *	</ul>
   *
   * Postconditions: <ul>
   *	<li> <tt>this->full_range() == false</tt>
   * <li> <tt>this->size() == ubound - lbound + 1</tt>
   * <li> <tt>this->lbound() == lbound</tt>
   * <li> <tt>this->ubound() == ubound</tt>
   *	</ul>
   */
  inline Range1D(Index lbound, Index ubound);

  /** Returns \c true if the range represents the entire region (constructed
   * from \c Range1D())
   */
  inline bool full_range() const;

  /** \brief Return lower bound of the range */
  inline Index lbound() const;

  /** \brief Return upper bound of the range */
  inline Index ubound() const;

  /** \brief Return the size of the range (<tt>ubound() - lbound() + 1</tt>) */
  inline Index size() const;

  /** \brief Return true if the index is in range */
  inline bool in_range(Index i) const;

  /** \brief Increment the range by a constant */
  inline Range1D& operator+=( Index incr );

  /** \brief  Deincrement the range by a constant */
  inline Range1D& operator-=( Index incr );

private:
  Index lbound_;
  Index ubound_;	// = INT_MAX-1 flag for entire range
  // lbound == ubound == 0 flag for invalid range.
  
  // assert that the range is valid
  inline void assert_valid_range(Index lbound, Index ubound) const;
  
}; // end class Range1D
  
/** \brief rng1 == rng2.
 *
 * @return Returns <tt>rng1.lbound() == rng2.ubound() && rng1.ubound() == rng2.ubound()</tt>.
 *
 * \relates Range1D
 */
inline bool operator==(const Range1D& rng1, const Range1D& rng2 )
{
  return rng1.lbound() == rng2.lbound() && rng1.ubound() == rng2.ubound();
}

/** \brief rng_lhs = rng_rhs + i.
  *
  * Increments the upper and lower bounds by a constant.
  *
  * Postcondition: <ul>
  *	<li> <tt>rng_lhs.lbound() == rng_rhs.lbound() + i</tt>
  *	<li> <tt>rng_lhs.ubound() == rng_rhs.ubound() + i</tt>
  *	</ul>
 *
 * \relates Range1D
  */
inline Range1D operator+(const Range1D &rng_rhs, Range1D::Index i)
{
    return Range1D(i+rng_rhs.lbound(), i+rng_rhs.ubound());
}

/** \brief rng_lhs = i + rng_rhs.
  *
  * Increments the upper and lower bounds by a constant.
  *
  * Postcondition: <ul>
  *	<li> <tt>rng_lhs.lbound() == i + rng_rhs.lbound()</tt>
  *	<li> <tt>rng_lhs.ubound() == i + rng_rhs.ubound()</tt>
  *	</ul>
 *
 * \relates Range1D
  */
inline Range1D operator+(Range1D::Index i, const Range1D &rng_rhs)
{
    return Range1D(i+rng_rhs.lbound(), i+rng_rhs.ubound());
}

/** \brief rng_lhs = rng_rhs - i.
  *
  * Deincrements the upper and lower bounds by a constant.
  *
  * Postcondition: <ul>
  *	<li> <tt>rng_lhs.lbound() == rng_rhs.lbound() - i</tt>
  *	<li> <tt>rng_lhs.ubound() == rng_rhs.ubound() - i</tt>
  *	</ul>
 *
 * \relates Range1D
  */
inline Range1D operator-(const Range1D &rng_rhs, Range1D::Index i)
{
    return Range1D(rng_rhs.lbound()-i, rng_rhs.ubound()-i);
}

/** \brief Return a bounded index range from a potentially unbounded index
  * range.
  * 
  * Return a index range of lbound to ubound if rng.full_range() == true
  * , otherwise just return a copy of rng.
  *
  * Postconditions: <ul>
  *	<li> [<tt>rng.full_range() == true</tt>] <tt>return.lbound() == lbound</tt>
  *	<li> [<tt>rng.full_range() == true</tt>] <tt>return.ubound() == ubound</tt>
  *	<li> [<tt>rng.full_range() == false</tt>] <tt>return.lbound() == rng.lbound()</tt>
  *	<li> [<tt>rng.full_range() == false</tt>] <tt>return.ubound() == rng.ubound()</tt>
  *	</ul>
 *
 * \relates Range1D
  */
inline Range1D full_range(const Range1D &rng, Range1D::Index lbound, Range1D::Index ubound)
{	return rng.full_range() ? Range1D(lbound,ubound) : rng; }

// //////////////////////////////////////////////////////////
// Inline members

inline
Range1D::Range1D()
  : lbound_(0), ubound_(INT_MAX-1)
{}

inline
Range1D::Range1D( EInvalidRange )
  : lbound_(0), ubound_(-1)
{}


inline
Range1D::Range1D(Index lbound, Index ubound)
  : lbound_(lbound), ubound_(ubound)
{
  assert_valid_range(lbound,ubound);
}

inline
bool Range1D::full_range() const {
  return ubound_ == INT_MAX-1;
}

inline
Range1D::Index Range1D::lbound() const {
  return lbound_;
}

inline
Range1D::Index Range1D::ubound() const {
  return ubound_;
}

inline
Range1D::Index Range1D::size() const {
  return ubound_ - lbound_ + 1;
}

inline
bool Range1D::in_range(Index i) const {
  return lbound_ <= i && i <= ubound_;
}

inline
Range1D& Range1D::operator+=( Index incr ) {
  assert_valid_range( lbound_ + incr, ubound_ + incr );
  lbound_ += incr;
  ubound_ += incr;
  return *this;
}

inline
Range1D& Range1D::operator-=( Index incr ) {
  assert_valid_range( lbound_ - incr, ubound_ - incr );
  lbound_ -= incr;
  ubound_ -= incr;
  return *this;
}

// See Range1D.cpp
inline
void Range1D::assert_valid_range(int lbound, int ubound) const {
#ifdef TEUCHOS_DEBUG
  TEST_FOR_EXCEPTION(
    lbound < 0, std::range_error
    ,"Range1D::assert_valid_range(): Error, lbound ="<<lbound<<" must be greater than or equal to 0."
    );
  TEST_FOR_EXCEPTION(
    lbound > ubound, std::range_error
    ,"Range1D::assert_valid_range(): Error, lbound = "<<lbound<<" > ubound = "<<ubound
    );
#endif
}

} // end namespace Teuchos

#endif // end TEUCHOS_RANGE1D_HPP
