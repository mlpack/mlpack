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

// $Id: SPCachedFactory.h 810 2007-02-25 14:36:28Z syntheticpp $

#ifndef SPCACHEDFACTORY_H_
#define SPCACHEDFACTORY_H_

/**
 * This file is intented to be used if you want a CachedFactory with
 * a SmartPointer encapsulation policy.
 * It as been defined in a separate file because of the many introduced
 * dependencies (SmartPtr.h would depend on Functor.h and CachedFactory.h
 * would depend on SmartPtr.h). By defining another header you pay for those
 * extra dependencies only if you need it. 
 * 
 * This file defines FunctionStorage a new SmartPointer storage policy and
 * SmartPointer a new CachedFactory encapsulation policy.
 */

#include <loki/Functor.h>
#include <loki/SmartPtr.h>
#include <loki/CachedFactory.h>

namespace Loki
{

////////////////////////////////////////////////////////////////////////////////
///  \class FunctionStorage
///
///  \ingroup  SmartPointerStorageGroup 
///  \brief Implementation of the StoragePolicy used by SmartPtr.
///
///  This storage policy is used by SmartPointer CachedFactory's encapsulation
///  policy. It's purpose is to call a Functor instead of deleting the
///  underlying pointee object. You have to set the callback functor by calling
///  SetCallBackFunction(const FunctorType &functor).
///
///  Unfortunately, the functor argument is not a reference to the SmartPtr but
///  a void *. Making functor argument a reference to the pointer would require
///  the FunctionStorage template to know the full definition of the SmartPtr.
////////////////////////////////////////////////////////////////////////////////

    template <class T>
    class FunctionStorage
    {
    public:
    	/// the type of the pointee_ object
        typedef T* StoredType;
        /// type used to declare OwnershipPolicy type.
        typedef T* InitPointerType;
        /// type returned by operator->
        typedef T* PointerType;
        /// type returned by operator*
        typedef T& ReferenceType;
        /// type of the Functor to set
        typedef Functor< void , Seq< void* > > FunctorType;

        FunctionStorage() : pointee_(Default()), functor_()
        {}

        // The storage policy doesn't initialize the stored pointer 
        //     which will be initialized by the OwnershipPolicy's Clone fn
        FunctionStorage(const FunctionStorage& rsh) : pointee_(0), functor_(rsh.functor_)
        {}

        template <class U>
        FunctionStorage(const FunctionStorage<U>& rsh) : pointee_(0), functor_(rsh.functor_)
        {}
        
        FunctionStorage(const StoredType& p) : pointee_(p), functor_() {}
        
        PointerType operator->() const { return pointee_; }
        
        ReferenceType operator*() const { return *pointee_; }
        
        void Swap(FunctionStorage& rhs)
        { 
        	std::swap(pointee_, rhs.pointee_);
        	std::swap(functor_, rhs.functor_);
        }
        
        /// Sets the callback function to call. You have to specify it or
        /// the smartPtr will throw a bad_function_call exception.
        void SetCallBackFunction(const FunctorType &functor)
        {
        	functor_ = functor;
        }
    
        // Accessors
        template <class F>
        friend typename FunctionStorage<F>::PointerType GetImpl(const FunctionStorage<F>& sp);

        template <class F>
        friend const typename FunctionStorage<F>::StoredType& GetImplRef(const FunctionStorage<F>& sp);

        template <class F>
        friend typename FunctionStorage<F>::StoredType& GetImplRef(FunctionStorage<F>& sp);

    protected:
        // Destroys the data stored
        // (Destruction might be taken over by the OwnershipPolicy)
        void Destroy()
        {
            functor_(this);
        }

        // Default value to initialize the pointer
        static StoredType Default()
        { return 0; }
    
    private:
        // Data
        StoredType pointee_;
        FunctorType functor_;
    };

    template <class T>
    inline typename FunctionStorage<T>::PointerType GetImpl(const FunctionStorage<T>& sp)
    { return sp.pointee_; }

    template <class T>
    inline const typename FunctionStorage<T>::StoredType& GetImplRef(const FunctionStorage<T>& sp)
    { return sp.pointee_; }

    template <class T>
    inline typename FunctionStorage<T>::StoredType& GetImplRef(FunctionStorage<T>& sp)
    { return sp.pointee_; }

    /**
	 * \class	SmartPointer
	 * \ingroup	EncapsulationPolicyCachedFactoryGroup
	 * \brief	Encapsulate the object in a SmartPtr with FunctionStorage policy.
	 * 
	 * The object will come back to the Cache as soon as no more SmartPtr are
	 * referencing this object. You can customize the SmartPointer with the standard
	 * SmartPtr policies (OwnershipPolicy, ConversionPolicy, CheckingPolicy,
	 * ConstnessPolicy) but StoragePolicy is forced to FunctionStorage.
	 */
     template
     <
     	class AbstractProduct,
     	template <class> class OwnershipPolicy = RefCounted,
        class ConversionPolicy = DisallowConversion,
        template <class> class CheckingPolicy = AssertCheck,
        template<class> class ConstnessPolicy = LOKI_DEFAULT_CONSTNESS 
     >     
     class SmartPointer
     {
     private:
     	   typedef SmartPtr< AbstractProduct,OwnershipPolicy,
     	   	ConversionPolicy, CheckingPolicy,
     	   	FunctionStorage, ConstnessPolicy > CallBackSP;
     protected:           
           typedef CallBackSP ProductReturn;
           SmartPointer() : fun(this, &SmartPointer::smartPointerCallbackFunction) {}
           virtual ~SmartPointer(){};
           
           ProductReturn encapsulate(AbstractProduct* pProduct)
           {
           		CallBackSP SP(pProduct);
           		SP.SetCallBackFunction(fun);
                return SP;
           }
           
           AbstractProduct* release(ProductReturn &pProduct)
           {
                return GetImpl(pProduct);
           }
           
           const char* name(){return "smart pointer";}

     private:
           SmartPointer& operator=(const SmartPointer&);
           SmartPointer(const SmartPointer&);
     	   void smartPointerCallbackFunction(void* pSP)
     	   {
     	   		CallBackSP &SP(*reinterpret_cast<CallBackSP*>(pSP));
     	   		ReleaseObject(SP);
     	   }
           virtual void ReleaseObject(ProductReturn &object)=0;
           const typename CallBackSP::FunctorType fun;
     };

} // namespace Loki

#endif /*SPCACHEDFACTORY_H_*/
