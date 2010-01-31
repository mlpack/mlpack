#ifndef YASLI_TRAITS_H_
#define YASLI_TRAITS_H_

// $Id: yasli_traits.h 754 2006-10-17 19:59:11Z syntheticpp $




namespace yasli_nstd 
{
    /*
    template <bool b, class T = void> 
    struct enable_if {};

    template <class T> 
    struct enable_if<true, T> { typedef T type; };
    */
    
    //!! TYPE SELECTORS
    //Used in place of enable_if:
    //not so neat or so versitile but they do compile
    template<bool condition, class if_true, class if_false>
    struct type_selector
    {
        typedef if_true result; 
    };
    
    template<class if_true, class if_false>
    struct type_selector<false, if_true, if_false>
    {
        typedef if_false result; 
    };  

    // Types for differentiating compile-time choices
    typedef char (&yes_t)[1];
    typedef char (&no_t)[2];

    // Credit goes to Boost; 
    // also found in the C++ Templates book by Vandevoorde and Josuttis

    //!! Wouldn't compile with these inside is_class
    template <class U>
    yes_t class_test(int U::*);
    template <class U>
    no_t class_test(...);

    template <class T> struct is_class
    {
        enum { value = (sizeof(class_test<T>(0)) == sizeof(yes_t)) };
    };

    template <typename T> struct is_pointer
    {
        enum { value = false };
    };

    template <typename T> struct is_pointer<T*>
    {
        enum { value = true };
    };

    template <typename T> struct is_memcopyable
    {
        enum { value = int(!is_class<T>::value) };
    };

    
   template <typename T> struct is_memmoveable
    {
        enum { value = int(!is_class<T>::value) };
    };
   

    // For moving
    enum move_t { move };

} // namespace yasli_nstd

#endif // YASLI_TRAITS_H_
