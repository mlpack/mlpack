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

// Kris
// 06.18.03 -- Minor formatting changes
//          -- Changed calls to LAPACK objects to use new <OType, SType> templates
// 07.08.03 -- Move into Teuchos package/namespace
// 07.11.03 -- Added ScalarTraits for ARPREC::mp_real
// 07.14.03 -- Fixed int rand() function (was set up to return a floating-point style random number)
// 07.17.03 -- Added squareroot() function

#ifndef _TEUCHOS_SCALARTRAITS_HPP_
#define _TEUCHOS_SCALARTRAITS_HPP_

/*! \file Teuchos_ScalarTraits.hpp
    \brief Defines basic traits for the scalar field type.
*/
 
#include "Teuchos_ConfigDefs.hpp"

#ifndef HAVE_NUMERIC_LIMITS
#include "Teuchos_LAPACK.hpp"
#endif

#ifdef HAVE_TEUCHOS_ARPREC
#include "mp/mpreal.h"
#endif

#ifdef HAVE_TEUCHOS_GNU_MP
#include "gmp.h"
#include "gmpxx.h"
#endif

/*! \struct Teuchos::ScalarTraits
    \brief This structure defines some basic traits for a scalar field type.

    Scalar traits are an essential part of templated codes.  This structure offers
    the basic traits of the templated scalar type, like defining zero and one,
    and basic functions on the templated scalar type, like performing a square root.

    The functions in the templated base unspecialized struct are designed not to
    compile (giving a nice compile-time error message) and therefore specializations
    must be written for Scalar types actually used.

    \note 
     <ol>
       <li> The default defined specializations are provided for \c int, \c float, and \c double.
     	 <li> ScalarTraits can be used with the Arbitrary Precision Library ( \c http://crd.lbl.gov/~dhbailey/mpdist/ )
            by configuring Teuchos with \c --enable-teuchos-arprec and giving the appropriate paths to ARPREC.
            Then ScalarTraits has the specialization: \c mp_real.
     	 <li> If Teuchos is configured with \c --enable-teuchos-std::complex then ScalarTraits also has
            a parital specialization for all std::complex numbers of the form <tt>std::complex<T></tt>.
     </ol>
*/

/* This is the default structure used by ScalarTraits<T> to produce a compile time
	error when the specialization does not exist for type <tt>T</tt>.
*/
namespace Teuchos {

template<class T>
struct UndefinedScalarTraits
{
  //! This function should not compile if there is an attempt to instantiate!
  static inline T notDefined() { return T::this_type_is_missing_a_specialization(); }
};

template<class T>
struct ScalarTraits
{
  //! Madatory typedef for result of magnitude
  typedef T magnitudeType;
  //! Determines if scalar type is std::complex
  static const bool isComplex = false;
  //! Determines if scalar type supports relational operators such as <, >, <=, >=.
  static const bool isComparable = false;
  //! Determines if scalar type have machine-specific parameters (i.e. eps(), sfmin(), base(), prec(), t(), rnd(), emin(), rmin(), emax(), rmax() are supported)
  static const bool hasMachineParameters = false;
  //! Returns relative machine precision.
  static inline magnitudeType eps()   { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns safe minimum (sfmin), such that 1/sfmin does not overflow.
  static inline magnitudeType sfmin() { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns the base of the machine.
  static inline magnitudeType base()  { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns \c eps*base.
  static inline magnitudeType prec()  { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns the number of (base) digits in the mantissa.
  static inline magnitudeType t()     { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns 1.0 when rounding occurs in addition, 0.0 otherwise
  static inline magnitudeType rnd()   { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns the minimum exponent before (gradual) underflow.
  static inline magnitudeType emin()  { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns the underflow threshold - \c base^(emin-1)
  static inline magnitudeType rmin()  { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns the largest exponent before overflow.
  static inline magnitudeType emax()  { return UndefinedScalarTraits<T>::notDefined(); }
  //! Overflow theshold - \c (base^emax)*(1-eps)
  static inline magnitudeType rmax()  { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns the magnitudeType of the scalar type \c a.
  static inline magnitudeType magnitude(T a) { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns representation of zero for this scalar type.
  static inline T zero()                     { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns representation of one for this scalar type.
  static inline T one()                      { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns the real part of the scalar type \c a.
  static inline magnitudeType real(T a) { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns the imaginary part of the scalar type \c a.
  static inline magnitudeType imag(T a) { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns the conjugate of the scalar type \c a.
  static inline T conjugate(T a) { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns a number that represents NaN.
  static inline T nan()                      { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns <tt>true</tt> if <tt>x</tt> is NaN or Inf.
  static inline bool isnaninf(const T& x)     { return UndefinedScalarTraits<T>::notDefined(); }
  //! Seed the random number generator returned by <tt>random()</tt>.
  static inline void seedrandom(unsigned int s) { int i; T t = &i; }
  //! Returns a random number (between -one() and +one()) of this scalar type.
  static inline T random()                   { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns the name of this scalar type.
  static inline std::string name()           { (void)UndefinedScalarTraits<T>::notDefined(); return 0; }
  //! Returns a number of magnitudeType that is the square root of this scalar type \c x. 
  static inline T squareroot(T x) { return UndefinedScalarTraits<T>::notDefined(); }
  //! Returns the result of raising one scalar \c x to the power \c y.
  static inline T pow(T x, T y) { return UndefinedScalarTraits<T>::notDefined(); }
};
  
#ifndef DOXYGEN_SHOULD_SKIP_THIS


void throwScalarTraitsNanInfError( const std::string &errMsg );


#define TEUCHOS_SCALAR_TRAITS_NAN_INF_ERR( VALUE, MSG ) \
  if (isnaninf(VALUE)) { \
    std::ostringstream omsg; \
    omsg << MSG; \
    throwScalarTraitsNanInfError(omsg.str()); \
  }


template<>
struct ScalarTraits<char>
{
  typedef char magnitudeType;
  static const bool isComplex = false;
  static const bool isComparable = true;
  static const bool hasMachineParameters = false;
  // Not defined: eps(), sfmin(), base(), prec(), t(), rnd(), emin(), rmin(), emax(), rmax()
  static inline magnitudeType magnitude(char a) { return static_cast<char>(std::fabs(static_cast<double>(a))); }
  static inline char zero()  { return 0; }
  static inline char one()   { return 1; }
  static inline char conjugate(char x) { return x; }
  static inline char real(char x) { return x; }
  static inline char imag(char x) { return 0; }
  static inline void seedrandom(unsigned int s) { std::srand(s); }
  //static inline char random() { return (-1 + 2*rand()); } // RAB: This version should be used to be consistent with others
  static inline char random() { return std::rand(); } // RAB: This version should be used for an unsigned char, not char
  static inline std::string name() { return "char"; }
  static inline char squareroot(char x) { return (char) std::sqrt((double) x); }
  static inline char pow(char x, char y) { return (char) std::pow((double)x,(double)y); }
};

template<>
struct ScalarTraits<int>
{
  typedef int magnitudeType;
  static const bool isComplex = false;
  static const bool isComparable = true;
  static const bool hasMachineParameters = false;
  // Not defined: eps(), sfmin(), base(), prec(), t(), rnd(), emin(), rmin(), emax(), rmax()
  static inline magnitudeType magnitude(int a) { return static_cast<int>(std::fabs(static_cast<double>(a))); }
  static inline int zero()  { return 0; }
  static inline int one()   { return 1; }
  static inline int conjugate(int x) { return x; }
  static inline int real(int x) { return x; }
  static inline int imag(int x) { return 0; }
  static inline void seedrandom(unsigned int s) { std::srand(s); }
  //static inline int random() { return (-1 + 2*rand()); }  // RAB: This version should be used to be consistent with others
  static inline int random() { return std::rand(); }             // RAB: This version should be used for an unsigned int, not int
  static inline std::string name() { return "int"; }
  static inline int squareroot(int x) { return (int) std::sqrt((double) x); }
  static inline int pow(int x, int y) { return (int) std::pow((double)x,(double)y); }
};

#ifndef __sun
extern const float flt_nan;
#endif
 
template<>
struct ScalarTraits<float>
{
  typedef float magnitudeType;
  static const bool isComplex = false;
  static const bool isComparable = true;
  static const bool hasMachineParameters = true;
  static inline float eps()   {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<float>::epsilon();
#else
    LAPACK<int, float> lp; return lp.LAMCH('E');
#endif
  }
  static inline float sfmin() {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<float>::min();
#else
    LAPACK<int, float> lp; return lp.LAMCH('S');
#endif
  }
  static inline float base()  {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<float>::radix;
#else
    LAPACK<int, float> lp; return lp.LAMCH('B');
#endif
  }
  static inline float prec()  {
#ifdef HAVE_NUMERIC_LIMITS
    return eps()*base();
#else
    LAPACK<int, float> lp; return lp.LAMCH('P');
#endif
  }
  static inline float t()     {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<float>::digits;
#else
    LAPACK<int, float> lp; return lp.LAMCH('N');
#endif
  }
  static inline float rnd()   {
#ifdef HAVE_NUMERIC_LIMITS
    return ( std::numeric_limits<float>::round_style == std::round_to_nearest ? float(1.0) : float(0.0) );
#else
    LAPACK<int, float> lp; return lp.LAMCH('R');
#endif
  }
  static inline float emin()  {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<float>::min_exponent;
#else
    LAPACK<int, float> lp; return lp.LAMCH('M');
#endif
  }
  static inline float rmin()  {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<float>::min();
#else
    LAPACK<int, float> lp; return lp.LAMCH('U');
#endif
  }
  static inline float emax()  {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<float>::max_exponent;
#else
    LAPACK<int, float> lp; return lp.LAMCH('L');
#endif
  }
  static inline float rmax()  {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<float>::max();
#else
    LAPACK<int, float> lp; return lp.LAMCH('O');
#endif
  }
  static inline magnitudeType magnitude(float a)
    {
#ifdef TEUCHOS_DEBUG
      TEUCHOS_SCALAR_TRAITS_NAN_INF_ERR(
        a, "Error, the input value to magnitude(...) a = " << a << " can not be NaN!" );
#endif      
      return std::fabs(a);
    }    
  static inline float zero()  { return(0.0); }
  static inline float one()   { return(1.0); }    
  static inline float conjugate(float x)   { return(x); }    
  static inline float real(float x) { return x; }
  static inline float imag(float x) { return 0; }
  static inline float nan() {
#ifdef __sun
    return 0.0/std::sin(0.0);
#else
    return flt_nan;
#endif
  }
  static inline bool isnaninf(float x) { // RAB: 2004/05/28: Taken from NOX_StatusTest_FiniteValue.C
    const float tol = 1e-6; // Any (bounded) number should do!
    if( !(x <= tol) && !(x > tol) ) return true;                 // IEEE says this should fail for NaN
    float z=0.0*x; if( !(z <= tol) && !(z > tol) ) return true;  // Use fact that Inf*0 = NaN
    return false;
  }
  static inline void seedrandom(unsigned int s) { std::srand(s); }
  static inline float random() { float rnd = (float) std::rand() / RAND_MAX; return (float)(-1.0 + 2.0 * rnd); }
  static inline std::string name() { return "float"; }
  static inline float squareroot(float x)
    {
#ifdef TEUCHOS_DEBUG
      TEUCHOS_SCALAR_TRAITS_NAN_INF_ERR(
        x, "Error, the input value to squareroot(...) x = " << x << " can not be NaN!" );
#endif
      errno = 0;
      const float rtn = std::sqrt(x);
      if (errno)
        return nan();
      return rtn;
    }
  static inline float pow(float x, float y) { return std::pow(x,y); }
};

#ifndef __sun
extern const double dbl_nan;
#endif
 
template<>
struct ScalarTraits<double>
{
  typedef double magnitudeType;
  static const bool isComplex = false;
  static const bool isComparable = true;
  static const bool hasMachineParameters = true;
  static inline double eps()   {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<double>::epsilon();
#else
    LAPACK<int, double> lp; return lp.LAMCH('E');
#endif
  }
  static inline double sfmin() {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<double>::min();
#else
    LAPACK<int, double> lp; return lp.LAMCH('S');
#endif
  }
  static inline double base()  {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<double>::radix;
#else
    LAPACK<int, double> lp; return lp.LAMCH('B');
#endif
  }
  static inline double prec()  {
#ifdef HAVE_NUMERIC_LIMITS
    return eps()*base();
#else
    LAPACK<int, double> lp; return lp.LAMCH('P');
#endif
  }
  static inline double t()     {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<double>::digits;
#else
    LAPACK<int, double> lp; return lp.LAMCH('N');
#endif
  }
  static inline double rnd()   {
#ifdef HAVE_NUMERIC_LIMITS
    return ( std::numeric_limits<double>::round_style == std::round_to_nearest ? double(1.0) : double(0.0) );
#else
    LAPACK<int, double> lp; return lp.LAMCH('R');
#endif
  }
  static inline double emin()  {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<double>::min_exponent;
#else
    LAPACK<int, double> lp; return lp.LAMCH('M');
#endif
  }
  static inline double rmin()  {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<double>::min();
#else
    LAPACK<int, double> lp; return lp.LAMCH('U');
#endif
  }
  static inline double emax()  {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<double>::max_exponent;
#else
    LAPACK<int, double> lp; return lp.LAMCH('L');
#endif
  }
  static inline double rmax()  {
#ifdef HAVE_NUMERIC_LIMITS
    return std::numeric_limits<double>::max();
#else
    LAPACK<int, double> lp; return lp.LAMCH('O');
#endif
  }
  static inline magnitudeType magnitude(double a)
    {
#ifdef TEUCHOS_DEBUG
      TEUCHOS_SCALAR_TRAITS_NAN_INF_ERR(
        a, "Error, the input value to magnitude(...) a = " << a << " can not be NaN!" );
#endif      
      return std::fabs(a);
    }
  static inline double zero()  { return 0.0; }
  static inline double one()   { return 1.0; }
  static inline double conjugate(double x)   { return(x); }    
  static inline double real(double x) { return(x); }
  static inline double imag(double x) { return(0); }
  static inline double nan() {
#ifdef __sun
    return 0.0/std::sin(0.0);
#else
    return dbl_nan;
#endif
  }
  static inline bool isnaninf(double x) { // RAB: 2004/05/28: Taken from NOX_StatusTest_FiniteValue.C
    const double tol = 1e-6; // Any (bounded) number should do!
    if( !(x <= tol) && !(x > tol) ) return true;                  // IEEE says this should fail for NaN
    double z=0.0*x; if( !(z <= tol) && !(z > tol) ) return true;  // Use fact that Inf*0 = NaN
    return false;
  }
  static inline void seedrandom(unsigned int s) { std::srand(s); }
  static inline double random() { double rnd = (double) std::rand() / RAND_MAX; return (double)(-1.0 + 2.0 * rnd); }
  static inline std::string name() { return "double"; }
  static inline double squareroot(double x)
    {
#ifdef TEUCHOS_DEBUG
      TEUCHOS_SCALAR_TRAITS_NAN_INF_ERR(
        x, "Error, the input value to squareroot(...) x = " << x << " can not be NaN!" );
#endif      
      errno = 0;
      const double rtn = std::sqrt(x);
      if (errno)
        return nan();
      return rtn;
    }
  static inline double pow(double x, double y) { return std::pow(x,y); }
};

#ifdef HAVE_TEUCHOS_GNU_MP

extern gmp_randclass gmp_rng; 

template<>
struct ScalarTraits<mpf_class>
{
  typedef mpf_class magnitudeType;
  static const bool isComplex = false;
  static const bool isComparable = true;
  static const bool hasMachineParameters = false;
  // Not defined: eps(), sfmin(), base(), prec(), t(), rnd(), emin(), rmin(), emax(), rmax()
  static magnitudeType magnitude(mpf_class a) { return std::abs(a); }
  static inline mpf_class zero() { mpf_class zero = 0.0; return zero; }
  static inline mpf_class one() { mpf_class one = 1.0; return one; }    
  static inline mpf_class conjugate(mpf_class x) { return x; }
  static inline mpf_class real(mpf_class x) { return(x); }
  static inline mpf_class imag(mpf_class x) { return(0); }
  static inline bool isnaninf(mpf_class x) { return false; } // mpf_class currently can't handle nan or inf!
  static inline void seedrandom(unsigned int s) { 
    unsigned long int seedVal = static_cast<unsigned long int>(s);
    gmp_rng.seed( seedVal );	
  }
  static inline mpf_class random() { 
    return gmp_rng.get_f(); 
  }
  static inline std::string name() { return "mpf_class"; }
  static inline mpf_class squareroot(mpf_class x) { return std::sqrt(x); }
  static inline mpf_class pow(mpf_class x, mpf_class y) { return pow(x,y); }
  // Todo: RAB: 2004/05/28: Add nan() and isnaninf() functions when needed!
};

#endif  

#ifdef HAVE_TEUCHOS_ARPREC

template<>
struct ScalarTraits<mp_real>
{
  typedef mp_real magnitudeType;
  static const bool isComplex = false;
  static const bool isComparable = true;
  static const bool hasMachineParameters = false;
  // Not defined: eps(), sfmin(), base(), prec(), t(), rnd(), emin(), rmin(), emax(), rmax()
  static magnitudeType magnitude(mp_real a) { return std::abs(a); }
  static inline mp_real zero() { mp_real zero = 0.0; return zero; }
  static inline mp_real one() { mp_real one = 1.0; return one; }    
  static inline mp_real conjugate(mp_real x) { return x; }
  static inline mp_real real(mp_real x) { return(x); }
  static inline mp_real imag(mp_real x) { return(0); }
  static inline bool isnaninf(mp_real x) { return false; } // ToDo: Change this?
  static inline void seedrandom(unsigned int s) { 
    long int seedVal = static_cast<long int>(s);
    srand48(seedVal);
  }
  static inline mp_real random() { return mp_rand(); }
  static inline std::string name() { return "mp_real"; }
  static inline mp_real squareroot(mp_real x) { return std::sqrt(x); }
  static inline mp_real pow(mp_real x, mp_real y) { return pow(x,y); }
  // Todo: RAB: 2004/05/28: Add nan() and isnaninf() functions when needed!
};
  
#endif // HAVE_TEUCHOS_ARPREC
 
#if ( defined(HAVE_COMPLEX) || defined(HAVE_COMPLEX_H) ) && defined(HAVE_TEUCHOS_COMPLEX)

// Partial specialization for std::complex numbers templated on real type T
template<class T> 
struct ScalarTraits<
#if defined(HAVE_COMPLEX)
  std::complex<T>
#elif  defined(HAVE_COMPLEX_H)
std::complex<T>
#endif
>
{
#if defined(HAVE_COMPLEX)
  typedef std::complex<T>  ComplexT;
#elif  defined(HAVE_COMPLEX_H)
  typedef std::complex<T>     ComplexT;
#endif
  typedef typename ScalarTraits<T>::magnitudeType magnitudeType;
  static const bool isComplex = true;
  static const bool isComparable = false;
  static const bool hasMachineParameters = true;
  static inline magnitudeType eps()          { return ScalarTraits<magnitudeType>::eps(); }
  static inline magnitudeType sfmin()        { return ScalarTraits<magnitudeType>::sfmin(); }
  static inline magnitudeType base()         { return ScalarTraits<magnitudeType>::base(); }
  static inline magnitudeType prec()         { return ScalarTraits<magnitudeType>::prec(); }
  static inline magnitudeType t()            { return ScalarTraits<magnitudeType>::t(); }
  static inline magnitudeType rnd()          { return ScalarTraits<magnitudeType>::rnd(); }
  static inline magnitudeType emin()         { return ScalarTraits<magnitudeType>::emin(); }
  static inline magnitudeType rmin()         { return ScalarTraits<magnitudeType>::rmin(); }
  static inline magnitudeType emax()         { return ScalarTraits<magnitudeType>::emax(); }
  static inline magnitudeType rmax()         { return ScalarTraits<magnitudeType>::rmax(); }
  static magnitudeType magnitude(ComplexT a)
    {
#ifdef TEUCHOS_DEBUG
      TEUCHOS_SCALAR_TRAITS_NAN_INF_ERR(
        a, "Error, the input value to magnitude(...) a = " << a << " can not be NaN!" );
#endif      
      return std::abs(a);
    }
  static inline ComplexT zero()              { return ComplexT(ScalarTraits<magnitudeType>::zero(),ScalarTraits<magnitudeType>::zero()); }
  static inline ComplexT one()               { return ComplexT(ScalarTraits<magnitudeType>::one(),ScalarTraits<magnitudeType>::zero()); }
  static inline ComplexT conjugate(ComplexT a){ return ComplexT(a.real(),-a.imag()); }
  static inline magnitudeType real(ComplexT a) { return a.real(); }
  static inline magnitudeType imag(ComplexT a) { return a.imag(); }
  static inline ComplexT nan()               { return ComplexT(ScalarTraits<magnitudeType>::nan(),ScalarTraits<magnitudeType>::nan()); }
  static inline bool isnaninf(ComplexT x)    { return ScalarTraits<magnitudeType>::isnaninf(x.real()) || ScalarTraits<magnitudeType>::isnaninf(x.imag()); }
  static inline void seedrandom(unsigned int s) { ScalarTraits<magnitudeType>::seedrandom(s); }
  static inline ComplexT random()
    {
      const T rnd1 = ScalarTraits<magnitudeType>::random();
      const T rnd2 = ScalarTraits<magnitudeType>::random();
      return ComplexT(rnd1,rnd2);
    }
  static inline std::string name() { return std::string("std::complex<")+std::string(ScalarTraits<magnitudeType>::name())+std::string(">"); }
  // This will only return one of the square roots of x, the other can be obtained by taking its conjugate
  static inline ComplexT squareroot(ComplexT x)
    {
#ifdef TEUCHOS_DEBUG
      TEUCHOS_SCALAR_TRAITS_NAN_INF_ERR(
        x, "Error, the input value to squareroot(...) x = " << x << " can not be NaN!" );
#endif
      typedef ScalarTraits<magnitudeType>  STMT;
      const T r  = x.real(), i = x.imag();
      const T a  = STMT::squareroot((r*r)+(i*i));
      const T nr = STMT::squareroot((a+r)/2);
      const T ni = STMT::squareroot((a-r)/2);
      return ComplexT(nr,ni);
    }
  static inline ComplexT pow(ComplexT x, ComplexT y) { return pow(x,y); }
};

#endif //  HAVE_COMPLEX || HAVE_COMPLEX_H

#endif // DOXYGEN_SHOULD_SKIP_THIS

} // Teuchos namespace

#endif // _TEUCHOS_SCALARTRAITS_HPP_
