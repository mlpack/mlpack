/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file otrav.h
 *
 * Object-tree traversal for the purpose of deep-copying,
 * pretty-printing, and advanced memory management.
 *
 * @see namespace ot, OBJECT_TRAVERSAL
 */

#ifndef BASE_OTRAV_H
#define BASE_OTRAV_H

#include "common.h"
#include "debug.h"
#include "cc.h"
#include "ccmem.h"

#include <typeinfo>

/**
 * Inside OBJECT_TRAVERSAL (or, more likely, OT_CUSTOM_PRINT),
 * traverse an object or primitive, but giving it a specified name and
 * type to print.
 *
 * You do not need to use this macro even in OT_CUSTOM_PRINT unless it
 * is most convenient for x to be some big nasty expression or of a
 * type that is not true to its value.  Otherwise, just use OT_OBJ.
 *
 * Example (printing NULL for pointers):
 * @code
 *   class MyClass {
 *     index_t len_;
 *     double *array_;
 *     ...
 *     OT_CUSTOM_PRINT() {
 *       OT_OBJ(len_);
 *       if (!array_) {
 *         OT_CUSTOM_PRINT_OBJ("NULL", "array_", typeid(array_).name());
 *       } else {
 *         OT_ARRAY(array_, len_);
 *       }
 *       ...
 *     }
 *     ...
 *   };
 * @endcode
 *
 * @param x the object or value to traverse; in OT_CUSTOM_PRINT, may
 *        be any expression; otherwise must be an lvalue expression
 * @param name a string giving the object's printed name
 * @param type a string giving the object's printed type
 *
 * @see OT_OBJ, OT_CUSTOM_PRINT, OBJECT_TRAVERSAL
 */
#define OT_CUSTOM_PRINT_OBJ(x, name) \
    if (true) { \
      ot__visitor->Name(name, x); \
      ot__visitor->Obj(x); \
    } else NOP // require semicolon

/**
 * Inside OBJECT_TRAVERSAL, traverse a non-pointer object or
 * primitive.  This is usually the declaration you want.
 *
 * OT_OBJ is designed for use with primitives and other classes that
 * have defined OBJECT_TRAVERSAL, but it can work with untraversed
 * classes.  Nearly all FASTlib core classes are traversable.
 *
 * You should use OT_OBJ for pointers that your class is not
 * responsible for deallocating, such as string constants.  These will
 * be shallow-copied and initialized to BIG_BAD_POINTER, but not
 * freed.  If you intend to serialize or freeze your class, you should
 * instead make sure to allocate copies for all pointers and arrays
 * and use OT_PTR, OT_ARRAY, or their expert versions.
 *
 * Example (composure with a vector):
 * @code
 *   class MyClass {
 *     Vector vec_;
 *     ...
 *     OBJECT_TRAVERSAL() {
 *       OT_OBJ(vec_);
 *       ...
 *     }
 *     ...
 *   };
 * @endcode
 *
 * @param x the object; may be any lvalue expression, e.g. @c array[i]
 *
 * @see OT_PTR, OT_ARRAY, OBJECT_TRAVERSAL
 */
#define OT_OBJ(x) \
    OT_CUSTOM_PRINT_OBJ(x, #x)

/**
 * Inside OT_ENUM_EXPERT (itself inside OBJECT_TRAVERSAL), test for
 * and handle a specific enum value when printing.
 *
 * @param val the enum value, as defined in @c enum{val,...};
 *
 * @see OT_ENUM_EXPERT
 */
#define OT_ENUM_VAL(val) \
   case val: \
    if (ot__visitor->IS_PRINTER) { \
      const char *ot__temp = #val; \
      ot__visitor->Obj(ot__temp); \
      break; \
    } // fall-through to default inside OT_ENUM_EXPERT

//    if (ot__visitor->IS_PRINTER && ot__enum == val) {
//      const char *ot__temp = #val;
//      ot__visitor->Obj(ot__temp);
//    } else // for series of OT_ENUM_VAL, use in OT_ENUM_EXPERT

/**
 * Inside OBJECT_TRAVERSAL, traverse an enum, optionally providing
 * a printable list of values that it might occupy.
 *
 * This macro exists because C++ templates cannot recognize enums as
 * primitives.  It is roughly equivalent to @c OT_OBJ((int)x) but does
 * a better job of printing the member name and type.
 *
 * Example (cannonical card suits):
 * @code
 *   class MyClass {
 *     enum Suit {
 *       CLUBS,
 *       SPADES,
 *       DIAMONDS,
 *       HEARTS
 *     };
 *     Suit suit_;
 *     ...
 *     OBJECT_TRAVERSAL() {
 *       OT_ENUM_EXPERT(suit_, int,
 *         OT_ENUM_VAL(CLUBS)
 *         OT_ENUM_VAL(SPADES)
 *         OT_ENUM_VAL(DIAMONDS)
 *         OT_ENUM_VAL(HEARTS));
 *       ...
 *     }
 *     ...
 *   };
 * @endcode
 *
 * Note that each OT_ENUM_VAL is white-space separated rather than
 * comma or semi-colon separated.  It is only necessary to provide the
 * list of OT_ENUM_VAL invocations if you want to print value names
 * rather than numbers, and only in the traversal function used by
 * ot::Print (i.e. if you have OT_CUSTOM_PRINT, your OBJECT_TRAVERSAL
 * is free to omit the list).  Omit the list by leaving only
 * white-space between the last comma and close parenthesis.
 *
 * Because this list is needed each time the enum appears in some
 * object's print traversal, it may be a good idea to define a macro
 * that exands to the list at the time that the enum is defined.
 *
 * @param x the enum; may be any lvalue expreesion, e.g. @c array[i]
 * @param T the enum's inherited type; almost certainly int
 * @param print_code a list of white-space separated OT_ENUM_VAL's
 *
 * @see OT_ENUM_VAL, OT_ENUM, OBJECT_TRAVERSAL
 */
#define OT_ENUM_EXPERT(x, T, print_code) \
    if (true) { \
      ot__visitor->Name(#x, x); \
      switch (x) { \
	print_code \
       default: \
	ot__visitor->Enum(x); \
      } \
    } else NOP // require semicolon

/**
 * Inside OBJECT_TRAVERSAL, traverse an enum as if it were an int.
 *
 * This macro exists because C++ templates cannot recognize enums as
 * primitives.  It is roughly equivalent to @c OT_OBJ((int)x) but does
 * a better job of printing the member name and type.
 *
 * Equivalent to @c OT_ENUM_EXPERT(x, int, )
 *
 * @param x the enum; may be any lvalue expresion, e.g. @c array[i]
 *
 * @see OT_ENUM_EXPERT, OT_OBJ, OBJECT_TRAVERSAL
 */
#define OT_ENUM(x) \
    OT_ENUM_EXPERT(x, int, )

/**
 * The implementation for OT_STATIC_ARRAY_EXPERT, OT_PTR_EXPERT,
 * OT_ARRAY_EXPERT, and OT_ALLOC_EXPERT.
 *
 * There is no need to invoke this macro unless you know what you're
 * doing.
 *
 * @param x the array; may be any lvalue expresion, e.g. @c *ptr
 * @param len the length of the array; may be any expression but must
 *        depend only on previously declared variables
 * @param func the visitor call used to handle the array; will be
 *        prefixed with Pre and Post for before and aftet the loop
 * @param iter_var a variable created to iterate the array
 * @param loop_code another member declaration, e.g. OT_OBJ
 *
 * @see OT_STATIC_ARRAY_EXPERT, OT_PTR_EXPERT, OT_ARRAY_EXPERT,
 *      OT_ALLOC_EXPERT, OBJECT_TRAVERSAL
 */
#define OT_ARRAY_IMPL(x, len, iter_var, loop_code, func, args...) \
    if (true) { \
      ot__visitor->Name(#x, x); \
      index_t ot__len = len; \
      if (ot__visitor->Pre##func(x, ot__len, ## args)) { \
	for (index_t iter_var = 0; iter_var < ot__len; ++iter_var) { \
	  ot__visitor->ElemOf(x); \
	  loop_code; \
	} \
        ot__visitor->Name(#x, x); \
	ot__visitor->Post##func(x, ot__len, ## args); \
      } \
    } else NOP // require semicolon

/**
 * Inside OBJECT_TRAVERSAL, traverse a static array, i.e. with size
 * given at compile time, and its contents.
 *
 * Example (static array of pointers):
 * @code
 *   class MyClass {
 *     MyClass *children_[NUM_CHILDREN];
 *     ...
 *     OBJECT_TRAVERSAL(MyClass} {
 *       OT_STATIC_ARRAY_EXPERT(children_,
 *         i, OT_PTR(children_[i]));
 *       ...
 *     }
 *     ...
 *   };
 * @endcode
 *
 * @param x the array; may be any lvalue expresion, e.g. @c *ptr
 * @param iter_var a variable created to iterate the array
 * @param elem_code another member declaration, e.g. OT_OBJ
 *
 * @see OT_STATIC_ARRAY, OBJECT_TRAVERSAL
 */
#define OT_STATIC_ARRAY_EXPERT(x, iter_var, elem_code) \
    OT_ARRAY_IMPL(x, sizeof(x) / sizeof(*x), \
        iter_var, elem_code, StaticArray)

/**
 * Inside OBJECT_TRAVERSAL, traverse a static array, i.e. with size
 * given at compile time, assuming its contents to be objects or
 * primitives.
 *
 * Equivalent to @c OT_STATIC_ARRAY_EXPERT(x, i, OT_OBJ(x[i]))
 *
 * @param x the array; may be any lvalue expression, e.g. @c *ptr
 *
 * @see OT_STATIC_ARRAY_EXPERT, OT_ARRAY, OBJECT_TRAVERSAL
 */
#define OT_STATIC_ARRAY(x) \
    OT_STATIC_ARRAY_EXPERT(x, ot__iter, OT_OBJ((x)[ot__iter]))

/**
 * Inside OBJECT_TRAVERSAL, traverse a pointer managed by new and
 * delete and its contents.
 *
 * Example (non-null pointer, for maximal speed):
 * @code
 *   class MyClass {
 *     MyTree *tree_;
 *     ...
 *     OBJECT_TRAVERSAL(MyClass} {
 *       OT_PTR_EXPERT(tree_, false,
 *         OT_OBJ(*tree_));
 *       ...
 *     }
 *     ...
 *   };
 * @endcode
 *
 * You should instead use OT_OBJ for pointers that your class is not
 * responisble for deallocating, especially string constants and other
 * const pointers.
 *
 * @param x the pointer; may be any lvalue expresion, e.g. @c array[i]
 * @param nullable bool indicating whether the pointer might be NULL
 * @param deref_code another member declaration, e.g. OT_OBJ
 *
 * @see OT_PTR, OBJECT_TRAVERSAL
 */
#define OT_PTR_EXPERT(x, nullable, deref_code) \
    OT_ARRAY_IMPL(x, 1, ot__iter, deref_code, \
        Array, nullable, false, true)

/**
 * Inside OBJECT_TRAVERSAL, traverse a pointer managed by new and
 * delete, assuming its contents to be an object or primitive.
 *
 * Equivalent to @c OT_PTR_EXPERT(x, true, OT_OBJ(*x))
 *
 * You should instead use OT_OBJ for pointers that your class is not
 * responisble for deallocating, especially string constants and other
 * const pointers.
 *
 * @param x the pointer; may be any lvalue expresion, e.g. @c array[i]
 *
 * @see OT_PTR_EXPERT, OT_ARRAY, OT_OBJ, OBJECT_TRAVERSAL
 */
#define OT_PTR(x) \
    OT_PTR_EXPERT(x, true, OT_OBJ(*(x)))

/**
 * Inside OBJECT_TRAVERSAL, traverse an array managed by new[] and
 * delete[] and its contents.
 *
 * Example (two-dimensional array):
 * @code
 *   class MyClass {
 *     index_t rows_;
 *     index_t cols_;
 *     double **data_;
 *     ...
 *     OBJECT_TRAVERSAL(MyClass} {
 *       OT_OBJ(rows_);
 *       OT_OBJ(cols_);
 *       OT_ARRAY_EXPERT(array_, rows_, true,
 *         i, OT_ARRAY(array_[i], cols));
 *       ...
 *     }
 *   ...
 *   };
 * @endcode
 *
 * For character strings (or any array), make sure you explicitly
 * allocate buffers when initializing/modifying members declared with
 * this macro.  These members will be freed, and will thus segfault if
 * they are set to string constants.  You may use OT_OBJ for arrays
 * that should not be freed.
 *
 * Note that, even when x is not nullable, x == NULL will be managed
 * properly if len == 0.  After copying, it may be that x != NULL, but
 * if so, it is safe to free/delete x.  Your code should never expect
 * non-nullable arrays to be NULL, but instead should assume they
 * always point to freeable/deletable memory or NULL.  (Note that it
 * is safe to free/delete NULL.)
 *
 * @param x the array; may be any lvalue expresion, e.g. @c *ptr
 * @param len the length of the array; may be any expression but must
 *        depend only on previously declared variables
 * @param nullable bool indicating whether the array might be NULL
 * @param iter_var a variable created to iterate the array
 * @param elem_code another member declaration, e.g. OT_OBJ
 *
 * @see OT_ARRAY, OBJECT_TRAVERSAL
 */
#define OT_ARRAY_EXPERT(x, len, nullable, iter_var, elem_code) \
    OT_ARRAY_IMPL(x, len, iter_var, elem_code, \
        Array, nullable, false, false)

/**
 * Inside OBJECT_TRAVERSAL, traverse an array managed by new[] and
 * delete[], assuming its contents to be objects or primitives.
 *
 * Equivalent to @c OT_ARRAY_EXPERT(x, len, true, i, OT_OBJ(x[i]))
 *
 * For character strings (or any array), make sure you explicitly
 * allocate buffers when initializing/modifying members declared with
 * this macro.  These members will be freed, and will thus segfault if
 * they are set to string constants.  You may use OT_OBJ for arrays
 * that should not be freed.
 *
 * @param x the array; may be any lvalue expression, e.g. @c *ptr
 * @param len the length of the array; may be any expression but must
 *        depend only on previously declared variables
 *
 * @see OT_ARRAY_EXPERT, OT_STATIC_ARRAY, OT_ALLOC, OBJECT_TRAVERSAL
 */
#define OT_ARRAY(x, len) \
    OT_ARRAY_EXPERT(x, len, true, ot__iter, OT_OBJ((x)[ot__iter]))

/**
 * Inside OBJECT_TRAVERSAL, traverse an array managed by mem::Alloc
 * and mem::Free and its contents.
 *
 * Example (array of mem::Alloc'd pointers):
 * @code
 *   class MyClass {
 *     index_t num_children_;
 *     MyClass **children_;
 *     ...
 *     OBJECT_TRAVERSAL(MyClass} {
 *       OT_OBJ(num_children_);
 *       OT_ALLOC_EXPERT(children_, num_children_, true,
 *         i, OT_ALLOC(children_[i], 1));
 *       ...
 *     }
 *     ...
 *   };
 * @endcode
 *
 * This function is also suitable for pointers--i.e. singleton
 * arrays--managed by mem::Alloc and mem::Free.
 *
 * @param x the array; may be any lvalue expresion, e.g. @c *ptr
 * @param len the length of the array; may be any expression but must
 *        depend only on previously declared variables
 * @param nullable bool indicating whether the array might be NULL
 * @param iter_var a variable created to iterate the array
 * @param elem_code another member declaration, e.g. OT_OBJ
 *
 * @see OT_ALLOC, OBJECT_TRAVERSAL
 */
#define OT_ALLOC_EXPERT(x, len, nullable, iter_var, elem_code) \
    OT_ARRAY_IMPL(x, len, iter_var, elem_code, \
        Array, nullable, true, false)

/**
 * Inside OBJECT_TRAVERSAL, traverse an array managed by mem::Alloc
 * and mem::Free, assuming its contents to be objects or primitives.
 *
 * Equivalent to @c OT_ALLOC_EXPERT(x, len, true, i, OT_OBJ(x[i]))
 *
 * This function is also suitable for pointers--i.e. singleton
 * arrays--managed by mem::Alloc and mem::Free.
 *
 * @param x the array; may be any lvalue expression, e.g. @c *ptr
 * @param len the length of the array; may be any expression but must
 *        depend only on previously declared variables
 *
 * @see OT_ALLOC_EXPERT, OT_ARRAY, OBJECT_TRAVERSAL
 */
#define OT_ALLOC(x, len) \
    OT_ALLOC_EXPERT(x, len, true, ot__iter, OT_OBJ((x)[ot__iter]))

/**
 * Inside OBJECT_TRAVERSAL, traverse a null-terminated character
 * string.
 *
 * This macro should only be used for strings with explicitly
 * allocated buffers.  These strings will be freed, and will thus
 * segfault if they are set to string constants.  (It is safe to use
 * this macro with string constants in OT_CUSTOM_PRINT, though OT_OBJ
 * may be used in either case.)
 *
 * When copying, freezing, and serializing, traversal allocates only
 * as much memory as is needed to store the string.  Accordingly,
 * while it is always safe to shorten the string by setting some
 * character to '\0', it is not generally safe to assume memory is
 * available beyond the terminating null, even if your Init functions
 * leave extra space.  Further, because length must be computed by
 * scanning the string, some traversal operations may be slower,
 * especially for large strings.
 *
 * For faster management of strings with explicit length, instead use
 * OT_ARRAY or OT_ALLOC.  This macro is perfered only when it is
 * inconvenient or unecessary to store the buffer's size separately.
 *
 * @param x the string; may be any lvalue expression, e.g. @c *ptr
 * @param nullable bool indicating whether the string might be NULL
 * @param alloc bool indicating whether to use mem::Alloc (true) or
 *        new (false)
 *
 * @see OT_STR, OT_ARRAY_EXPERT, OT_ALLOC_EXPERT, OT_OBJ,
 *      OBJECT_TRAVERSAL
 */
#define OT_STR_EXPERT(x, nullable, alloc) \
    if (true) { \
      ot__visitor->Name(#x, x); \
      ot__visitor->Str(x, nullable, alloc); \
    } else NOP // require semicolon

/**
 * Inside OBJECT_TRAVERSAL, traverse a null-terminated character
 * string managed by mem::Alloc and mem::Free.
 *
 * Equivalent to @c OT_STR_EXPERT(x, true, true)
 *
 * @param x the string; may be any lvalue expression, e.g. @c *ptr
 *
 * @see OT_STR_EXPERT, OT_ALLOC, OT_OBJ, OBJECT_TRAVERSAL
 */
#define OT_STR(x) \
    OT_STR_EXPERT(x, true, true)



/**
 * Called by OBJECT_TRAVERSAL to define a default constructor for your
 * class.
 *
 * There is no need to invoke this macro unless you know what you're
 * doing.
 *
 * @param C the name of your class
 *
 * @see OT_DESTRUCTOR, OBJECT_TRAVERSAL
 */
#define OT_CONSTRUCTOR(C) \
   public: \
    C() { \
      ot__private::Unstructor ot__unstructor(this); \
      OT__DefaultConstruct(this); \
    } \
   private:

/**
 * Called by OBJECT_TRAVERSAL to define a destructor for your class.
 *
 * There is no need to invoke this macro unless you know what you're
 * doing.
 *
 * @param C the name of your class
 *
 * @see OT_CONSTRUCTOR, OBJECT_TRAVERSAL
 */
#define OT_DESTRUCTOR(C) \
   public: \
    ~C() { \
      DEBUG_DESTRUCT_OK(this); \
      ot__private::Destructor<false> ot__destructor(this); \
    } \
   private:

/**
 * Called by OBJECT_TRAVERSAL to define a copy constructor and
 * assignment for your class.  The copy constructor behaves
 * identically to ot::InitCopy, and assignment is defined in terms of
 * copy construction.
 *
 * There is no need to invoke this macro unless you know what you're
 * doing.
 *
 * @param C the name of your class
 *
 * @see ot::InitCopy, OT_COPY_METHOD, OBJECT_TRAVERSAL
 */
#define OT_COPY_CONSTRUCTOR(C) \
   public: \
    C(const C &src) { \
      ot__private::Copier<false> ot__copier(this, &src); \
    } \
    ASSIGN_VIA_RECURSION_SAFE_COPY_CONSTRUCTION(C) \
   private:

/**
 * Called by OBJECT_TRAVERSAL_DEPRECATED_COPIES to define a deprecated
 * copy constructor and assignment for your class.  This permites the
 * use of these functions, but generates compiler warnings so as to
 * encourage users to move towards OBJECT_TRAVERSAL_NO_COPES.
 *
 * There is no need to invoke this macro unless you know what you're
 * doing.
 *
 * @param C the name of your class
 *
 * @see ot::InitCopy, OT_COPY_METHOD, OBJECT_TRAVERSAL
 */
#define OT_DEPRECATED_COPY_CONSTRUCTOR(C) \
   public: \
    COMPILER_DEPRECATED \
    C(const C &src) { \
      ot__private::Copier<false> ot__copier(this, &src); \
    } \
    ASSIGN_VIA_RECURSION_SAFE_COPY_CONSTRUCTION(C) \
   private:



/**
 * Called by OBJECT_TRAVERSAL to define a Renew method for your class,
 * which calls the class's destructor followed by its constructor,
 * thereby preparing the object for reuse.
 *
 * There is no need to invoke this macro unless you know what you're
 * doing.
 *
 * @param C the name of your class
 *
 * @see OT_DESTRUCTOR, OT_CONSTRUCTOR, OBJECT_TRAVERSAL
 */
#define OT_RENEW_METHOD(C) \
   public: \
    void Renew() { \
      this->~C(); \
      new(this) C(); \
    } \
   private:

/**
 * Called by OBJECT_TRAVERSAL to define an InitCopy method for your
 * class, which behaves identically to ot::InitCopy and may be used in
 * the absense of a copy constructor.
 *
 * There is no need to invoke this macro unless you know what you're
 * doing.
 *
 * @param C the name of your class
 *
 * @see ot::InitCopy, OBJECT_TRAVERSAL
 */
#define OT_COPY_METHOD(C) \
   public: \
    void InitCopy(const C &src) { \
      ot::InitCopy(this, src); \
    } \
    COMPILER_DEPRECATED \
    void Copy(const C &src) { \
      ot::InitCopy(this, src); \
    } \
   private:

/**
 * Called by OBJECT_TRAVERSAL to define a Print method to your class,
 * which behaves identically to ot::Print, but having filled the
 * obvious parameter appropriately.
 *
 * There is no need to invoke this macro unless you know what you're
 * doing.
 *
 * @param C the name of your class
 *
 * @see ot::Print, OBJECT_TRAVERSAL
 */
#define OT_PRINT_METHOD(C) \
   public: \
    template<typename TPrintFormat> \
    const char *Print(const char *name, FILE *stream = stdout) { \
      return ot::Print<TPrintFormat>(*this, name, stream); \
    } \
    const char *Print(const char *name, FILE *stream = stdout) { \
      return Print<ot::StandardFormat>(name, stream); \
    } \
   private:



/**
 * Adds FrozenSize, Freeze, and InitThaw methods to your class, which
 * behave identically to ot::FrozenSize, ot::Freeze, and ot::InitThaw,
 * but having filled the obvious parameter appropriately.
 *
 * @param C the name of your class
 *
 * @see ot::FrozenSize, ot::Freeze, ot::InitThaw, OBJECT_TRAVERSAL
 */
#define OT_FREEZE_METHODS(C) \
   public: \
    size_t FrozenSize() { \
      return ot::FrozenSize(*this); \
    } \
    size_t Freeze(char *block) { \
      return ot::Freeze(block, *this); \
    } \
    void InitThaw(char *block) { \
      ot::InitThaw(this, block); \
    } \
   private:

/**
 * Adds SerialSize, Seralize, and InitDeserialize methods to your
 * class, which behave identically to ot::SerialSize, ot::Serialize,
 * and ot::InitDeserialize, but having filled the obvious parameter
 * appropriately.
 *
 * @param C the name of your class
 *
 * @see ot::SerialSize, ot::Serialize, ot::InitDeserialize,
 *      OBJECT_TRAVERSAL
 */
#define OT_SERIALIZE_METHODS(C) \
   public: \
    size_t SerialSize() { \
      return ot::SerialSize(*this); \
    } \
    size_t Serialize(FILE *stream) { \
      return ot::Serialize(*this, stream); \
    } \
    size_t InitDeserialize(FILE *stream) { \
      return ot::InitDeserialize(this, stream); \
    } \
   private:



/**
 * Fills a function to be called after all forms of copying to ensure
 * that copied objects are not considered aliases.  Code specified for
 * OT_BECOME_NON_ALIAS must cause code for OT_IS_ALIAS to return
 * false.
 *
 * This macro is unnecessary unless you also invoke OT_ALIAS_METHODS.
 * In addition, it is unnecessary if OT_REFILL_TRANSIENTS leaves the
 * object in a non-alias state.  Note that it is your Init functions'
 * responsibility to mark objects as non-aliases (or aliases, if
 * appropriate) when they are first initialized.  It is possible to
 * call @c OT__BecomeNonAlias_() explicitly.
 *
 * @param C the name of your class
 * @param following_code a block containing the code needed to
 *        put an object of your class into a non-alias state
 *
 * @see OT_ALIAS_METHODS, OT_BECOME_ALIAS, OT_IS_ALIAS,
 *      OBJECT_TRAVERSAL
 */
#define OT_BECOME_NON_ALIAS(C) \
   public: \
    friend void OT__BecomeNonAlias(C *ot__obj) { \
      ot__obj->OT__BecomeNonAlias_(); \
      DEBUG_ASSERT_MSG(!OT__IsAlias(ot__obj), \
          "OT_BECOME_NON_ALIAS left OT_IS_ALIAS true for %s.", \
	  typeid(C).name()); \
    } \
   private: \
    void OT__BecomeNonAlias_()

/**
 * Fills a function to be called during InitAlias and InitSteal to
 * ensure that aliased objects are considered aliases.  Code specified
 * for OT_BECOME_ALIAS must cause code for OT_IS_ALIAS to return true.
 *
 * This macro is unnecessary unless you also invoke OT_ALIAS_METHODS.
 * It is possible to call @c OT__BecomeAlias_() explicitly.
 *
 * @param C the name of your class
 * @param following_code a block containing the code needed to
 *        put an object of your class into an alias state
 *
 * @see OT_ALIAS_METHODS, OT_BECOME_NON_ALIAS, OT_IS_ALIAS,
 *      OBJECT_TRAVERSAL
 */
#define OT_BECOME_ALIAS(C) \
   public: \
    friend bool OT__Aliasable(const C *ot__obj) { \
      return true; \
    } \
    friend void OT__BecomeAlias(C *ot__obj) { \
      ot__obj->OT__BecomeAlias_(); \
      DEBUG_ASSERT_MSG(OT__IsAlias(ot__obj), \
          "OT_BECOME_ALIAS left OT_IS_ALIAS false for %s.", \
	  typeid(C).name()); \
    } \
   private: \
    void OT__BecomeAlias_()

/**
 * Fills a function to be called during destruction to decide whether
 * your object is responsible for its allocated memory.  Code
 * specified for OT_IS_ALIAS must return true after OT_BECOME_ALIAS an
 * false after OT_BECOME_NON_ALIAS.
 *
 * This macro is unnecessary unless you also invoke OT_ALIAS_METHODS.
 *
 * @param C the name of your class
 * @param following_code a block containing the code needed to
 *        test whether an object of your class is an alias
 * @returns whether the object is an alias
 *
 * @see OT_ALIAS_METHODS, OT_BECOME_NON_ALIAS, OT_BECOME_ALIAS,
 *      OBJECT_TRAVERSAL
 */
#define OT_IS_ALIAS(C) \
   public: \
    friend bool OT__IsAlias(C *ot__obj) { \
      return ot__obj->OT__IsAlias_(); \
    } \
   private: \
    bool OT__IsAlias_()

/**
 * Adds standard aliasing methods InitAlias, InitSteal, and IsAlias to
 * your class.  The Init methods behave similarly to InitCopy, but
 * create an alias and convert the source into an alias, respectively.
 *
 * This macro declaration must be used in conjunction with
 * OT_BECOME_NON_ALIAS, OT_BECOME_ALIAS, and OT_IS_ALIAS as well as
 * OBJECT_TRAVERSAL or one of its bretheren.
 *
 * Aliases behave exactly the same as their non-alias counterparts,
 * but do not free allocated memory.  Aliases are invalidated if
 * either they or the original copy are modified except below some
 * level of pointer indirection.  E.g. it is safe to modify the
 * contents of an aliased array, but resizing the array invalidates
 * all aliases.  Unfortunately, traversal cannot detect or enforce
 * invalidation; it is recommended that you assert @c !IsAlias()
 * before any modifications are made to an aliasable object.
 *
 * When copied, serialized, or frozen, aliases are converted into true
 * copies.  Accordingly, cyclical aliasing is forbidden except via
 * transients.  You can alias an alias, which has the same affect as
 * aliasing the original again.  To preserve aliasing relationships
 * after copying, it may be necessary to instead use transients.
 * (TODO: write a guide for advanced aliasing with object traversal.)
 *
 * Example (using an alias flag):
 * @code
 *   class MyClass {
 *     bool alias_var_;
 *     ...
 *     OT_BECOME_NON_ALIAS(MyClass) {
 *       alias_var_ = false;
 *     }
 *     OT_BECOME_ALIAS(MyClass) {
 *       alias_var_ = true;
 *     }
 *     OT_IS_ALIAS(MyClass) {
 *       return alias_var_;
 *     }
 *     OT_ALIAS_METHODS(MyClass);
 *     ...
 *   };
 * @endcode
 *
 * @param C the name of your class
 *
 * @see OT_BECOME_NON_ALIAS, OT_BECOME_ALIAS, OT_IS_ALIAS,
 *      OBJECT_TRAVERSAL
 */
#define OT_ALIAS_METHODS(C) \
   public: \
    void InitAlias(const C &src) { \
      DEBUG_INIT_OK(this); \
      mem::Copy(this, &src); \
      ot__private::Aliaser ot__aliaser(this); \
    } \
    void InitSteal(C *src) { \
      DEBUG_INIT_OK(this); \
      DEBUG_WARN_MSG_IF(OT__IsAlias(src), \
          "Stealing from an alias %s.", typeid(C).name()); \
      mem::Copy(this, src); \
      ot__private::Aliaser ot__aliaser(src); \
    } \
    bool IsAlias() { \
      return OT__IsAlias(this); \
    } \
   private:



/**
 * Fills a function to be called just after default construction.
 *
 * If aliasing is defined for your class, the default behavior of this
 * function is to flag freshly constructed objects as aliases;
 * otherwise, it does nothing.  This makes sense because aliases are
 * destructed trivially, (by default) may be overwritten by standard
 * Init functions, and (by default) may not be changed otherwise.
 *
 * If OT_BECOME_ALIAS cannot operate on freshly constructed objects,
 * you must invoke this macro with, for instance, {} to disable this
 * behavior.  Otherwise, you need only invoke this macro if your class
 * needs special construction on top of being poisoned and pointers
 * nullified.
 *
 * In any case, default construction must always leave objects such
 * that the code given for OT_DEBUG_INIT_OK returns true.
 *
 * Example (initializing to a non-alias):
 * @code
 *   class MyClass {
 *     bool is_alias_;
 *     ...
 *     OT_DEFAULT_CONSTRUCT(MyClass) {
 *       is_alias_ = false;
 *     }
 *     ...
 *   };
 * @endcode
 *
 * @param C the name of your class
 * @param following_code a block containing default construction code
 *
 * @see OT_BECOME_ALIAS, OT_DEBUG_INIT_OK, OT_DEBUG_MODIFY_OK,
 *      OT_REFILL_TRANSIENTS, OBJECT_TRAVERSAL
 */
#define OT_DEFAULT_CONSTRUCT(C) \
   public: \
    friend void OT__DefaultConstruct(C *ot__obj) { \
      ot__obj->OT__DefaultConstruct_(); \
    } \
   private: \
    void OT__DefaultConstruct_()

/**
 * Fills a function to be called by the DEBUG_INIT_OK macro.  This
 * macro is in turn called by all standard Init functions to test for
 * reinitialization.
 *
 * If aliasing is defined for your class, the default behavior of this
 * function is to return whether the object is an alias; otherwise, it
 * returns true.  In conjunction with the default behavior for
 * OT_DEFAULT_CONSTRUCT, this ensures that freshly constructed objects
 * may always be initialized.  It also permits standard Init functions
 * to overwrite aliases, which have no destruction responsibilities.
 *
 * You may optionally invoke this marco to perform an arbitrary test.
 * The provided function should not have any side-effects, must return
 * true for freshly constructed objects, and should return false
 * otherwise.  It may assume that it is called only in debug-mode.
 *
 * Example (using an initialization flag):
 * @code
 *   class MyClass {
 *     bool initialized_;
 *     ...
 *     OT_DEFAULT_CONSTRUCT(MyClass) {
 *       DEBUG_ONLY(initialized_ = false);
 *     }
 *     OT_DEBUG_INIT_OK(MyClass) {
 *       return !initialized_;
 *     }
 *     ...
 *   };
 * @endcode
 *
 * @param C the name of your class
 * @param following_code a block containing the code needed to
 *        test whether an object of your class can be initialized
 * @returns whether the object can be initialized
 *
 * @see DEBUG_INIT_OK, OT_DEBUG_MODIFY_OK, OT_DEBUG_DESTRUCT_OK,
 *      OT_IS_ALIAS, OT_DEFAULT_CONSTRUCT, OBJECT_TRAVERSAL
 */
#define OT_DEBUG_INIT_OK(C) \
   public: \
    friend bool OT__DebugInitOK(C *ot__obj) { \
      return ot__obj->OT__DebugInitOK_(); \
    } \
   private: \
    bool OT__DebugInitOK_()

/**
 * Asserts in debug-mode that an object may be initialized, e.g. that
 * no Init function has been called.
 *
 * If aliasing is defined for your class, this macro (by default)
 * asserts that the object is an alias; otherwise, it does nothing.
 * Its behavior may be changed by invoking the OT_DEBUG_INIT_OK macro.
 *
 * @param x the object to assert uninitiailized
 *
 * @see OT_DEBUG_INIT_OK
 */
#define DEBUG_INIT_OK(x) \
    DEBUG_ASSERT_MSG(OT__DebugInitOK(x), \
        "Reinitialization of %s; missing Renew()?", \
        typeid(*x).name());

/**
 * Fills a function to be called by the DEBUG_MODIFY_OK macro, which
 * is meant to ensure that no changes are made by aliases that might
 * invalidate the originals.
 *
 * If aliasing is defined for your class, the default behavior of this
 * function is to return true if and only if the object is not an
 * alias; otherwise, it returns true.
 *
 * You may optionally invoke this marco to perform an arbitrary test.
 * The provided function should not have any side-effects.  It may
 * assume that it is called only in debug-mode.
 *
 * Example (using an lock flag):
 * @code
 *   class MyClass {
 *     bool locked_;
 *     ...
 *     OT_DEBUG_MODIFY_OK(MyClass) {
 *       return !locked_;
 *     }
 *     ...
 *   };
 * @endcode
 *
 * @param C the name of your class
 * @param following_code a block containing the code needed to
 *        test whether an object of your class can be modified
 * @returns whether the object can be modified
 *
 * @see DEBUG_MODIFY_OK, OT_DEBUG_INIT_OK, OT_DEBUG_DESTRUCT_OK,
 *      OT_IS_ALIAS, OBJECT_TRAVERSAL
 */
#define OT_DEBUG_MODIFY_OK(C) \
   public: \
    friend bool OT__DebugModifyOK(C *ot__obj) { \
      return ot__obj->OT__DebugModifyOK_(); \
    } \
   private: \
    bool OT__DebugModifyOK_()

/**
 * Asserts that an object may be modified, e.g. that it is not an
 * alias.
 *
 * If aliasing is defined for your class, this macro (by default)
 * asserts that the object is not an alias; otherwise, it does
 * nothing.  Its behavior may be changed by invoking the
 * OT_DEBUG_MODIFY_OK macro.
 *
 * @param x the object to assert modifiable
 *
 * @see OT_DEBUG_MODIFY_OK
 */
#define DEBUG_MODIFY_OK(x) \
    DEBUG_ASSERT_MSG(OT__DebugModifyOK(x), \
        "Modification of alias/locked %s; missing Init?", \
        typeid(*x).name());

/**
 * Fills a function to be called by the DEBUG_DESTRUCT_OK macro.  This
 * macro is in turn called by the object's destructor and Renew
 * method to test for premature destruction.
 *
 * By default, this function simply returns true.  You may optionally
 * invoke this marco to perform an arbitrary test.  The provided
 * function should not have any side-effects.  It may assume that it
 * is called only in debug-mode.
 *
 * Example (counting references):
 * @code
 *   class MyClass {
 *     bool n_references_;
 *     ...
 *     OT_DEBUG_DESTRUCT_OK(MyClass) {
 *       return n_references_ == 0;
 *     }
 *     ...
 *   };
 * @endcode
 *
 * @param C the name of your class
 * @param following_code a block containing the code needed to
 *        test whether an object of your class can be destroyed
 * @returns whether the object can be destroyed
 *
 * @see DEBUG_DESTRUCT_OK, OT_DEBUG_INIT_OK, OT_DEBUG_MODIFY_OK
 *      OBJECT_TRAVERSAL
 */
#define OT_DEBUG_DESTRUCT_OK(x) \
   public: \
    friend bool OT__DebugDestructOK(C *ot__obj) { \
      return ot__obj->OT__DebugDestructOK_(); \
    } \
   private: \
    bool OT__DebugDestructOK_()

/**
 * Asserts that an object may be destroyed.
 *
 * This macro does nothing unless you invoke the OT_DEBUG_DESTRUCT_OK
 * macro for your class.
 *
 * @param x the object to assert destructable
 *
 * @see OT_DEBUG_DESTRUCT_OK
 */
#define DEBUG_DESTRUCT_OK(x) \
    DEBUG_ASSERT_MSG(OT__DebugDestructOK(x), \
        "Premature destruction of %s; missing Init?", \
        typeid(*x).name());



/**
 * Fills a function to be called just after copy construction,
 * InitCopy, and other copying functions.  This is useful, for
 * example, for resetting transient pointers to their new contexts.
 *
 * Note that your own Init functions should either fill transients
 * themselves or call @c OT__RefillTransients_() explicitly.
 *
 * You do not need to invoke this macro if you have no transients.
 *
 * Example (setting a parent pointer):
 * @code
 *   class MyClass {
 *     MyClass *parent_;
 *     MyClass *child_;
 *     ...
 *     OBJECT_TRAVERSAL(MyClass) {
 *       OT_PTR(child_);
 *       ...
 *     }
 *     OBJECT_TRAVERSAL(MyClass) {
 *       OT_OBJ(parent_);
 *     }
 *     OT_REFILL_TRANSIENTS(MyClass) {
 *       parent_ = NULL;
 *       if (child_) {
 *         child_->parent_ = this;
 *       }
 *     }
 *     ...
 *   };
 * @endcode
 *
 * @param C the name of your class
 * @param following_code a block containing the code needed to
 *        construct your class's transients after a copy
 *
 * @see OT_DEFAULT_CONSTRUCT, OT_TRANSIENTS, OBJECT_TRAVERSAL
 */
#define OT_REFILL_TRANSIENTS(C) \
   public: \
    friend void OT__RefillTransients(C *ot__obj) { \
      OT__BecomeNonAlias(ot__obj); \
      ot__obj->OT__RefillTransients_(); \
    } \
   private: \
    void OT__RefillTransients_()

/**
 * Fills a function used to traverse your transients to debug-poison
 * them on default construction and properly deallocate them on
 * destruction.
 *
 * This function is declared similarly to OBJECT_TRAVERSAL.  All of
 * your class's members should occur in one of OBJECT_TRAVERSAL or
 * OT_TRANSIENTS.  Destruction of transients occurs before that of
 * other members, meaning, e.g., transient arrays may have
 * non-transient lengths but not vice versa.
 *
 * Example (declaring a parent pointer):
 * @code
 *   class MyClass {
 *     double *transient_array_;
 *     MyClass *parent_;
 *     ...
 *     OT_TRANSIENTS(MyClass) {
 *       OT_ARRAY(transient_array_, len_);
 *       OT_OBJ(parent_);
 *     }
 *     ...
 *   };
 * @endcode
 *
 * As you can see, some care is required to discriminate between
 * transient pointers that should be freed and those that reference
 * other memory.  In the latter case, it is appropriate to use OT_OBJ.
 *
 * While it rarely matters, transients must be declared in destruction
 * order.  For example, if some transient array's length is stored in
 * an allocated transient pointer, the array should be declared before
 * the pointer.  Note that this is the opposite of OBJECT_TRAVERSAL.
 *
 * @param C the name of your class
 * @param following_code a block containing macro declarations for
 *        your class's transient members; e.g. OT_OBJ
 *
 * @see OT_DEFAULT_CONSTRUCT, OT_REFILL_TRANSIENTS, OBJECT_TRAVERSAL
 */
#define OT_TRANSIENTS(C) \
   public: \
    template<typename TVisitor> \
    friend void OT__TraverseTransients(C *ot__obj, TVisitor *ot__visitor) { \
      ot__obj->OT__TraverseTransients_(ot__visitor); \
    } \
   private: \
    template<typename TVisitor> \
    void OT__TraverseTransients_(TVisitor *ot__visitor)



/**
 * OBJECT_TRAVERSAL_CORE calls this to give your class the bare-minium
 * for object traversal functionality.
 *
 * There is no need to invoke this macro unless you know what you're
 * doing.
 *
 * @param C the name of your class
 * @param following_code a block containing macro declarations for
 *        your class's members; e.g. OT_OBJ
 *
 * @see OBJECT_TRAVERSAL_CORE, OBJECT_TRAVERSAL
 */
#define OBJECT_TRAVERSAL_ONLY(C) \
   public: \
    template<typename TVisitor> \
    friend bool OT__PreTraverse(C *ot__dest, const C *ot__src, \
                                index_t ot__len, TVisitor *ot__visitor) { \
      return ot__visitor->PreTraverse(ot__dest, ot__src, ot__len); \
    } \
    template<typename TVisitor> \
    friend void OT__TraverseObject(C *ot__obj, TVisitor *ot__visitor) { \
      ot__obj->OT__TraverseObject_(ot__visitor); \
    } \
   private: \
    template<typename TVisitor> \
    void OT__TraverseObject_(TVisitor *ot__visitor)

/**
 * OBJECT_TRAVERSAL calls this to give your class traveral
 * functionality and to provide standard Copy and Print methods.
 *
 * You can invoke this macro instead of OBJECT_TRAVERSAL if you need
 * to define your own default constuctor, destructor, or copy
 * constructor and assignment, but otherwise you should use
 * OBJECT_TRAVERSAL or OBJECT_TRAVERSAL_NO_COPIES.
 *
 * @param C the name of your class
 * @param following_code a block containing macro declarations for
 *        your class's members; e.g. OT_OBJ
 *
 * @see OBJECT_TRAVERSAL, OBJECT_TRAVERSAL_NO_COPIES
 */
#define OBJECT_TRAVERSAL_CORE(C) \
    OT_RENEW_METHOD(C) \
    OT_COPY_METHOD(C) \
    OT_PRINT_METHOD(C) \
    OBJECT_TRAVERSAL_ONLY(C)

/**
 * Fills a function to provide all standard FASTlib functionality:
 * safe, debug-poisoned construction and destruction; pointer- and
 * array-allocating copy construction and assignment; standard Copy
 * and Print methods; and the use of other features in the ot
 * namespace including serialization and freezing.
 *
 * This macro establishes your class as a FASTlib-complient data
 * storage class and should be used whenever (reasonably) possible.
 * In order to use object traversal features, your class's overall
 * structure must have the following properties:
 *
 * @li No cyclical pointers that cannot be modeled as transients
 * @li No polymorphic inheritance or virtual functions
 * @li Minimally specialied constructors and destructors
 * @li Use untraversed member types at your own risk
 *
 * Example (custom linked list):
 * @code
 *   class MyClass {
 *     MyInfo info_;
 *     index_t data_len_;
 *     double *data_;
 *     MyClass *next_;
 *     ...
 *     OBJECT_TRAVERSAL(MyClass) {
 *       OT_OBJ(info_);
 *       OT_OBJ(data_len_);
 *       OT_ARRAY(data_, data_len_);
 *       OT_PTR(next_);
 *       ...
 *     }
 *     ...
 *   };
 * @endcode
 *
 * Traversal occurs in the order that members are declared in
 * OBJECT_TRAVERSAL; accordingly, array lengths must be filled before
 * they are used.  Also, note that the content you provide to
 * OBJECT_TRAVERSAL is just a function and you can run arbitrary code
 * in it, though this is only appropraite in rare circumstances such
 * as computing lengths for interestingly shaped arrays.
 *
 * Additional or alternate functions can be specified for certain
 * traversals.  Notably, macros OT_DEFAULT_TRANSIENTS,
 * OT_CONSTRUCT_TRANSIENTS, and OT_DESTRUCT_TRANSIENTS permit your
 * class to have fields not explicitly copied but instead inferred
 * from other parameters.  These can reconstruct parent pointers,
 * self-referential aliases, and even, for instance, pointers
 * representing an arbitrary graph's edge list.  Also, OT_CUSTOM_PRINT
 * allows for different behavior when pretty-printing, and
 * OT_CUSTOM_DESTRUCT is needed in rare occasions to safely deallocate
 * your class's memory.
 *
 * Keep in mind that not all classes need to invoke OBJECT_TRAVERSAL.
 * Some things just aren't meant to be copied, such as threads.
 *
 * @param C the name of your class
 * @param following_code a block containing macro declarations for
 *        your class's members; e.g. OT_OBJ
 *
 * @see OBJECT_TRAVERSAL_NO_COPIES, OBJECT_TRAVERSAL_AND_ALIAS,
 *      OBJECT_TRAVERSAL_SHALLOW, OT_OBJ, OT_PTR, OT_ARRAY,
 *      OT_CUSTOM_PRINT, OT_CUSTOM_DESTRUCT, OT_DEFAULT_TRANSIENTS,
 *      OT_CONSTRUCT_TRANSIENTS, OT_DESTRUCT_TRANSIENTS
 */
#define OBJECT_TRAVERSAL(C) \
    OT_CONSTRUCTOR(C) \
    OT_DESTRUCTOR(C) \
    OT_COPY_CONSTRUCTOR(C) \
    OBJECT_TRAVERSAL_CORE(C)

/**
 * Fills a function to provide all standard FASTlib functionality, as
 * with OBJECT_TRAVERSAL, but disables copy construnction and
 * assignment to prevent accidents.
 *
 * Your class may still be copied via the InitCopy method,
 * ot::InitCopy, and traversal-based copying of containing classes.
 *
 * @param C the name of your class
 *
 * @see OBJECT_TRAVERSAL
 */
#define OBJECT_TRAVERSAL_NO_COPIES(C) \
    OT_CONSTRUCTOR(C) \
    OT_DESTRUCTOR(C) \
    FORBID_ACCIDENTAL_COPIES(C) \
    OBJECT_TRAVERSAL_CORE(C)

/**
 * Fills a function to provide all standard FASTlib functionality, as
 * with OBJECT_TRAVERSAL, but flags copy construnction and assignment
 * as deprecated.  This permites the use of these functions, but
 * generates compiler warnings so as to encourage users to move
 * towards OBJECT_TRAVERSAL_NO_COPES.
 *
 * Your class may still be copied via the InitCopy method,
 * ot::InitCopy, and traversal-based copying of containing classes.
 *
 * @param C the name of your class
 *
 * @see OBJECT_TRAVERSAL
 */
#define OBJECT_TRAVERSAL_DEPRECATED_COPIES(C) \
    OT_CONSTRUCTOR(C) \
    OT_DESTRUCTOR(C) \
    OT_DEPRECATED_COPY_CONSTRUCTOR(C) \
    OBJECT_TRAVERSAL_CORE(C)

/**
 * Fills a traversal function, as with OBJECT_TRAVERSAL, but
 * establishes that your class is shallow, potentially speeding up
 * some traversal tasks.
 *
 * Shallow classes must have the following properties:
 * @li No owned pointers, i.e. nothing to deallocate
 * @li No transients to be filled on copy (and no aliasing)
 * @li All composed objects are also shallow
 *
 * Naturally, all non-pointer primitives are shallow, as are all
 * classes that contain only non-pointer primitives.  Traversal is
 * accelerated most notably when it can bit-copy entire arrays of
 * shallow objects.
 *
 * Shallow classes use default C++ copy constructors and assignment
 * rather than anything defined by object traversal.  Further, their
 * constructors and destructors simply debug-poison contained memory.
 *
 * @param C the name of your class
 * @param following_code a block containing macro declarations for
 *        your class's members; e.g. OT_OBJ
 *
 * @see OBJECT_TRAVERSAL
 */
#define OBJECT_TRAVERSAL_SHALLOW(C) \
   public: \
    C() { \
      mem::DebugPoison(this); \
    } \
    ~C() { \
      DEBUG_DESTRUCT_OK(this); \
      mem::DebugPoison(this); \
    } \
    friend bool OT__Shallow(const C *ot__obj) { \
      return true; \
    } \
   private: \
    OT_IS_ALIAS(C) { return false; } \
    OT_DEFAULT_CONSTRUCT(C) {} \
    OT_REFILL_TRANSIENTS(C) {} \
    OT_TRANSIENTS(C) {} \
    OBJECT_TRAVERSAL_CORE(C)



/**
 * Redefines traversal in the special case of ot::Print or the Print
 * method.  This is useful if normal traversal necessary for
 * successful copying, serialization, etc. is not particularly
 * human-readable.
 *
 * You may emit transients via OT_CUSTOM_PRINT by including them as
 * normal with OT_OBJ and its bretheren, but be careful not to create
 * cycles by traversing pointers to parents.  Further, you may emit
 * the results of functions or expressions.
 *
 * Example (printing a managed string):
 * @code
 *   class MyClass {
 *     char *c_str() { ... }
 *     ...
 *     OT_CUSTOM_PRINT(MyClass) {
 *       OT_STR(c_str());
 *     }
 *     ...
 *   };
 * @endcode
 *
 * @param C the name of your class
 * @param following_code a block containing the code needed to
 *        print your class; may consist of, e.g., OT_OBJ
 *
 * @see OBJECT_TRAVERSAL, ot::Print
 */
#define OT_CUSTOM_PRINT(C) \
   private: \
    template<typename TPrintFormat> \
    void OT__TraverseObject_(ot__private::Printer<TPrintFormat> *ot__visitor)

/**
 * Redefines traversal in the special case of the destructor or
 * ot::SemiDestruct.  This is necessary if normal traversal would
 * deallocate memory needed for the proper destruction of other
 * members.
 *
 * Example (deleting an array of variable length arrays of pointers):
 * @code
 *   class MyClass {
 *     index_t rows_;
 *     index_t *cols_;
 *     MyClass ***children_;
 *     ...
 *     OBJECT_TRAVERSAL(MyClass) {
 *       OT_OBJ(rows_);
 *       OT_ARRAY(cols_, rows_);
 *       OT_ARRAY_EXPERT(children_, rows_, true,
 *         i, OT_ARRAY_EXPERT(children_[i], cols_[i], false,
 *           j, OT_PTR_EXPERT(children_[i][j], false,
 *             OT_OBJ(*children_[i][j]))));
 *     }
 *     OT_CUSTOM_DESTRUCT(MyClass) {
 *       OT_ARRAY_EXPERT(children_, rows_, true,
 *         i, OT_ARRAY_EXPERT(children_[i], cols_[i], false,
 *           j, OT_PTR_EXPERT(children_[i][j], false,
 *             OT_OBJ(*children_[i][j]))));
 *       OT_ARRAY(cols_, rows_);
 *       OT_OBJ(rows_);
 *     }
 *     ...
 *   };
 * @endcode
 *
 * Note that destruct code is typically "safe" when it is the reverse
 * of construction code.  Unfortunately, this is beyond the power of
 * OBJECT_TRAVERSAL's automation.  On the other hand, it takes really
 * contrived types to defeat normal traversal behavior.  Further, the
 * example could be rewritten to use ArrayLists or other types that
 * keep length and pointer allocated together.
 *
 * @param C the name of your class
 * @param following_code a block containing the code needed to
 *        destruct your class; may consist of, e.g., OT_OBJ
 *
 * @see OBJECT_TRAVERSAL, ot::Print
 */
#define OT_CUSTOM_DESTRUCT(C) \
   private: \
    template<bool t_semi> \
    void OT__TraverseObject_(ot__private::Destructor<t_semi> *ot__visitor)



#define NEED_OTRAV_IMPL
#include "otrav_impl.h"
#undef NEED_OTRAV_IMPL



/**
 * Object traversal utilities for printing, copying, freezing, and
 * serialization.
 *
 * These tools operate on all classes that invoke the OBJECT_TRAVERSAL
 * macro or an equivalent, which includes all core FASTlib data
 * structures.  Specifically, they provide:
 *
 * @li Pretty-printing for debugging or saving to XML
 * @li A generic copy routine that works on primitives, untraversed
 *     classes with copy constructors, and all traversed classes
 * @li Object "freezing", which copies the object's contents into a
 *     flat buffer for easy transport and fast reuse
 * @li Serialization and deserialization to and from file streams.
 *
 * You can incorporate many of these functions as methods of your
 * class with the OT_FREEZE_METHODS and OT_SERIALIZE_METHODS macors.
 * Default OBJECT_TRAVERSAL already gives your class Print and
 * InitCopy methods.
 *
 * @see OBJECT_TRAVERSAL, OT_FREEZE_METHODS, OT_SERIALIZE_METHODS
 */
namespace ot {
  /**
   * Pretty-print an object using a given print format.
   *
   * Print formats are classes used by the traversal printer to emit
   * objects in different ways.  At current, two print formats are
   * offered: ot::StandardFormat and ot::XMLFormat.
   *
   * You can define your own print format; it's interface must
   * include:
   *
   * @li A constructor that accepts a FILE *stream.
   * @li A templated function @c Untraversed(const T &obj)
   * @li For each base type, templated/overloaded functions
   *     @c Primitive(const char *name, index_t index,
   *                  const char *type, T val)
   * @li A templated Primitive function that works on pointers
   * @li A templated function
   *     @c Open(const char *name, index_t index,
   *             const char *type, const T &obj, index_t len = -1)
   * @li A templated function
   *     @c Close(const char *name, const T &obj)
   *
   * The Untraversed and Primitive functions are responsible for
   * printing their respective types.  Open and Close are called
   * before and after traversing the contents of objects, pointers,
   * and arrays.  Only allocated arrays will have nonnegative lengths,
   * and unallocated (NULL) arrays and pointers are handled by
   * Primitive.  See FOR_ALL_PRIMITIVES_DO to define many Primitive
   * functions with a single macro.
   *
   * @param obj the object to be printed
   * @param name the printed name attributed to the object
   * @param stream the output stream
   * @return name, permitting reference, for instance, within printf
   *
   * @see OBJECT_TRAVERSAL, OT_CUSTOM_PRINT
   */
  template<typename TPrintFormat, typename T>
  const char *Print(const T &obj, const char *name, FILE *stream = stdout) {
    ot__private::Printer<TPrintFormat> ot__printer(obj, name, stream);
    return name;
  }
  template<typename TPrintFormat, typename T>
  void Print(const T &obj, FILE *stream = stdout) {
    ot__private::Printer<TPrintFormat> ot__printer(obj, "_", stream);
  }

  /**
   * Pretty-print an object using the standard print format.
   *
   * Example (debug message):
   * @code
   *   DEBUG_ASSERT_MSG(node.parent_ != NULL, "NULL parent (see %s).",
   *       ot::Print(node, "faulty_node", stderr));
   * @endcode
   *
   * @param obj the object to be printed
   * @param name the printed name attributed to the object
   * @param stream the output stream
   * @return name, permitting reference, for instance, within printf
   *
   * @see OBJECT_TRAVERSAL, OT_CUSTOM_PRINT
   */
  template<typename T>
  const char *Print(const T &obj, const char *name, FILE *stream = stdout) {
    return Print<ot::StandardFormat>(obj, name, stream);
  }
  template<typename T>
  void Print(const T &obj, FILE *stream = stdout) {
    Print<ot::StandardFormat>(obj, stream);
  }



  /**
   * Copy an object into a specified destination.
   *
   * This function works for primitive types and untraversed types
   * with copy constructors by calling their copy construstors.  For
   * traversed types, copying behaves identically to the copy
   * constructor if it exists, but no copy constructor is required.
   *
   * This function does not allocate memory for the top level of the
   * copied object, permitting you to copy objects onto the stack or
   * into your own allocated buffers.
   *
   * @param dest an uninitialized object to receive the copy
   * @param src the object to be copied
   *
   * @see OBJECT_TRAVERSAL, OT_CONSTRUCT_TRANSIENTS
   */
  template<typename T>
  inline void InitCopy(T *dest, const T &src) {
    DEBUG_INIT_OK(dest);
    ot__private::Copier<false> ot__copier(dest, &src);
  }



  /**
   * Determine the buffer size needed to store a frozen copy of an
   * object.
   *
   * @param obj the object whose frozen size should be computed
   * @return the size of the object in bytes after freezing
   *
   * @see Freeze, OBJECT_TRAVERSAL
   */
  template<typename T>
  inline size_t FrozenSize(const T &obj) {
    ot__private::Freezer<true> ot__freezer(NULL, obj);
    return ot__freezer.size();
  }

  /**
   * Copy an object's contents into flat buffer for easy transport,
   * normalizing pointers to offsets into the buffer for fast reuse.
   *
   * Freezing is the prefered method for transmitting objects from
   * machine to machine, storing them in temporary files, and other
   * situations where speed is favored over efficient use of space.
   * The primary advantage over rote serialization is that
   * ot::SemiThaw permits (formerly) frozen objects to be used
   * in-place.  There is no need to allocate memory separately from
   * the buffer holding the semi-thawed copy, and ot::SemiFreeze can
   * quickly repackage the object for subsequent transmittion.
   *
   * Example (sending an object):
   * @code
   *   MyClass my_obj;
   *   ... // Initialize my_obj
   *   char *buf = mem::Alloc<char>(ot::FrozenSize(my_obj));
   *   ot::Freeze(buf, my_obj);
   *   ... // Transmit over the network
   *   mem::Free(buf);
   * @endcode
   *
   * Example (receiving an object):
   * @code
   *   char *buf;
   *   ... // Fill buf from network
   *   MyClass *thawed_obj = ot::SemiThaw(buf);
   *   ... // Use thawed_obj
   *   ot::SemiDestruct(thawed_obj);
   *   mem::Free(buf);
   * @endcode
   *
   * To be more specific, ot::Freeze forms frozen objects by
   * bit-copying the data structure into a flat (e.g. contiguous)
   * buffer.  Traversed pointers and arrays are processed depth-first
   * and normalized to the bit-offsets of their represented regions
   * from the beginning of the buffer.  It is always safe to bit-copy
   * frozen objects to new locations and their containing buffers may
   * be freed without additional memory maintenance.
   *
   * Afterward, ot::SemiThaw converts a frozen object into a
   * semi-object by adding the buffer's memory location to all
   * contained pointers and reconstructing transients.  These are
   * fully operational up to the ability to allocate or free
   * non-transient fields.  It is possible to bit-copy semi-objects to
   * new locations, but after having done so, one must call either
   * ot::Relocate or ot::SemiCopy, the former simply updating pointer
   * positions (semantically invalidating the old location) and the
   * latter reconstructing transients as well (leaving the old
   * location valid).  Alternately, you may call ot::SemiFreeze to
   * revert the copy to a frozen state.  Semi-objects must be
   * explicitly destructed via ot::SemiDestruct, or may be frozen
   * in-place with ot::SemiFreezeDestruct.
   *
   * You can form a normal object directly from a frozen object with
   * ot::InitThaw.  Keep in mind that ot::InitCopy works when copying
   * from a semi-object.
   *
   * @param block the memory location to receive the frozen copy
   * @param src the object to be frozen
   * @return the size of the object in bytes after freezing
   *
   * @see FrozenSize, InitThaw, Relocate, SemiFreeze, SemiCopy,
   *      SemiThaw, SemiDestruct, SemiFreezeDestruct,
   *      OT_FREEZE_METHODS, Serialize, OBJECT_TRAVERSAL
   */
  template<typename T>
  inline size_t Freeze(char *block, const T &src) {
    ot__private::Freezer<false> ot__freezer(block, src);
    return ot__freezer.size();
  }

  /**
   * Copy from a frozen object into a specified destination.
   *
   * Has the same effect as ot::SemiThaw followed by ot::InitCopy.
   *
   * @param dest an uninitialized object to receive the copy
   * @param block the frozen object to be copied
   *
   * @see Freeze, SemiThaw, InitCopy, OBJECT_TRAVERSAL
   */
  template<typename T>
  inline void InitThaw(T *dest, const char *block) {
    DEBUG_INIT_OK(dest);
    ot__private::Copier<true> 
        ot__copier(dest, reinterpret_cast<const T *>(block));
  }



  /**
   * Fixes the pointers and reconstructs transients of a semi-object
   * assuming it has already been bit-copied into a new location.
   *
   * It is OK for the original location to have already been freed.
   *
   * @param block the new location of the semi-object to freeze
   * @param orig the original location of the semi-object
   * @return block recast to a T pointer
   *
   * @see Freeze, SemiThaw, Relocate, SemiFreeze
   */
  template<typename T>
  inline T *SemiCopy(char *block, const T *orig) {
    DEBUG_WARN_MSG_IF(block == reinterpret_cast<const char *>(orig),
	"In-place SemiCopy may leak memory; probably incorrect.");
    ot__private::Relocator<false, true> ot__relocator(block, orig);
    return reinterpret_cast<T *>(block);
  }
  template<typename T>
  inline T *SemiCopy(char *block, const char *orig) {
    return SemiCopy(block, reinterpret_cast<const T *>(orig));
  }

  /**
   * Re-freezes a semi-object assuming it has already been bit-copied
   * into a new location.
   *
   * Example (freezing a semi-object):
   * @code
   *   MyClass *semi_obj;
   *   char *freeze_buf;
   *   ... // Fill semi_obj with a semi-object of size bytes
   *   freeze_buf = mem::AllocCopyBytes<char>(semi_obj, bytes);
   *   ot::SemiFreeze(freeze_buf, semi_obj);
   *   ...
   *   ot::SemiDestruct(semi_obj);
   * @endcode
   *
   * You should not use this freeze semi-objects in place because this
   * precludes properly destructing their transients.  Instead,
   * bit-copy the semi-object to a new location and freeze it there,
   * or use ot::SemiFreezeDestruct.  It is OK for the original
   * location to have already been freed.
   *
   * @param block the new location of the semi-object to freeze
   * @param orig the original location of the semi-object
   *
   * @see Freeze, SemiFreezeDestruct, Relocate, SemiCopy
   */
  template<typename T>
  inline void SemiFreeze(char *block, const T *orig) {
    DEBUG_WARN_MSG_IF(block == reinterpret_cast<const char *>(orig),
        "In-place SemiFreeze may leak memory; use SemiFreezeDestruct.");
    ot__private::Relocator<true, false> ot__relocator(block, orig);
  }
  template<typename T>
  void SemiFreeze(char *block, const char *orig) {
    SemiFreeze(block, reinterpret_cast<const T *>(orig));
  }

  /**
   * Thaws a frozen object by fixing pointers and reconstructing
   * transients.
   *
   * Equivalent to ot::SemiCopy with orig equal to NULL.
   *
   * @param block the frozen object to be thawed
   * @return block recast to a T pointer
   *
   * @see Freeze, SemiFreeze
   */
  template<typename T>
  inline T *SemiThaw(char *block) {
    return SemiCopy(block, reinterpret_cast<const T *>(NULL));
  }

  /**
   * Destructs a semi-object's transients and debug-poisons its
   * memory.
   *
   * @param obj the semi-object to destroy
   *
   * @see Freeze, SemiFreezeDestruct
   */
  template<typename T>
  inline void SemiDestruct(T *obj) {
    ot__private::Destructor<true> ot__destructor(obj);
  }
  template<typename T>
  inline void SemiDestruct(char *block) {
    SemiDestruct(reinterpret_cast<T *>(block));
  }

  /**
   * Re-freezes a semi-object and destructs its transients.
   *
   * @param obj the semi-object to re-freeze
   * @return obj recast to a char pointer
   *
   * @see Freeze, SemiDestruct, SemiFreeze
   */
  template<typename T>
  inline char *SemiFreezeDestruct(T *obj) {
    ot__private::Relocator<true, true>
        ot__relocator(reinterpret_cast<char *>(obj), obj);
    return reinterpret_cast<char *>(obj);
  }
  template<typename T>
  inline char *SemiFreezeDestruct(char *block) {
    return SemiFreezeDestruct(reinterpret_cast<T *>(block));
  }



  /**
   * Determine the length in bytes of a serialized version of an
   * object.
   *
   * @param obj the object whose serial size should be computed
   * @return the size of the object in bytes after serialization
   *
   * @see Serialize, Deserialize, OBJECT_TRAVERSAL
   */
  template<typename T>
  inline size_t SerialSize(const T &obj) {
    ot__private::Serializer<true> ot__serializer(obj, NULL);
    return ot__serializer.size();
  }

  /**
   * Write a serialized version of an object to a file stream.
   *
   * Serialization is more compact than freezing because it emits an
   * object's members without the padding.  Further, it emits bools to
   * indicate whether or not a pointer is NULL (or nothing at all if
   * the pointer is not nullable), inferring the position of the
   * pointer's contents by depth-first traversal.  This violates the
   * alignment and stride constraints of various platforms, and thus
   * serialization is only available in the form of writing to and
   * reading from a file stream.
   *
   * Example (saving an object to file):
   * @code
   *   MyClass my_obj;
   *   ... // Initialize my_obj
   *   FILE *f_out = fopen("out.dat", "wb");
   *   ot::Serialize(my_obj, f_out);
   *   fclose(f_out);
   * @endcode
   *
   * @param obj the object to serialize
   * @param stream a file stream openned for binary output
   * @return number of bytes written to the stream
   *
   * @see Deserialize, SerialSize, Freeze, OBJECT_TRAVERSAL
   */
  template<typename T>
  inline size_t Serialize(const T &obj, FILE *stream) {
    ot__private::Serializer<false> ot__serializer(obj, stream);
    return ot__serializer.size();
  }

  /**
   * Read a serialized version of an object from a file stream,
   * reconstructing its transients.
   *
   * Example (reading an object from file):
   * @code
   *   MyClass my_obj;
   *   FILE *f_in = fopen("in.dat", "rb");
   *   ot::Deserialize(&my_obj, f_in);
   *   fclose(f_in);
   * @endcode
   *
   * @param dest an uninitialized object to receive the copy
   * @param stream a file stream openned for binary input
   * @return number of bytes read from the stream
   *
   * @see Serialize, SerialSize, Freeze, OBJECT_TRAVERSAL
   */
  template<typename T>
  inline size_t InitDeserialize(T *dest, FILE *stream) {
    DEBUG_INIT_OK(dest);
    ot__private::Deserializer ot__deserializer(dest, stream);
    return ot__deserializer.size();
  }



  /**
   * Equivalent to mem::Construct, but exploits properties of shallow
   * types.
   */
  template<typename T>
  inline T *Construct(T *array, size_t elems = 1) {
    if (OT__ShallowOrPtr(array)) {
      return mem::DebugPoison(array, elems);
    } else {
      return mem::Construct(array, elems);
    }
  }

  /**
   * Equivalent to mem::Destruct, but exploits properties of shallow
   * types.
   */
  template<typename T>
  inline T *Destruct(T *array, size_t elems = 1) {
    if (OT__ShallowOrPtr(array)) {
      return mem::DebugPoison(array, elems);
    } else {
      return mem::Destruct(array, elems);
    }
  }

  /**
   * Equivalent to mem::CopyConstruct, but exploits properties of
   * shallow types and does not require a copy constructor to be
   * defined for traversed types.
   *
   * Note the absense of DEBUG_INIT_OK.  This ensures that this
   * function works on arrays of unconstructed objects.
   */
  template<typename T>
  inline T *CopyConstruct(T *dest, const T *src, size_t elems = 1) {
    ot__private::Copier<false> ot__copier(dest, src, elems);
    return dest;
  }

  /**
   * Equivalent to mem::RepeatConstruct, but does not require a copy
   * constructor to be defined for traversed types.
   *
   * Note the absense of DEBUG_INIT_OK.  This ensures that this
   * function works on arrays of unconstructed objects.
   */
  template<typename T>
  inline T *RepeatConstruct(T *array, const T &init, size_t elems) {
    for (size_t i = 0; i < elems; ++i) {
      ot__private::Copier<false> ot__copier(array + i, &init);
    }
    return array;
  }

  ////////// Deprecated //////////////////////////////////////////////

  /** Renamed InitCopy */
  template<typename T>
  COMPILER_DEPRECATED
  void Copy(const T &src, T *dest) {
    InitCopy(dest, src);
  }

  /** Renamed FrozenSize */
  template<typename T>
  COMPILER_DEPRECATED
  size_t PointerFrozenSize(const T &obj) {
    return FrozenSize(obj);
  }

  /** Renamed Freeze */
  template<typename T>
  COMPILER_DEPRECATED
  void PointerFreeze(const T &obj, char *block) {
    Freeze(block, obj);
  }

  /** Renamed SemiFreeze */
  template<typename T>
  COMPILER_DEPRECATED
  void PointerRefreeze(const T *src, char *dest) {
    SemiFreeze(dest, src);
  }

  /** Renamed SemiThaw */
  template<typename T>
  COMPILER_DEPRECATED
  T *PointerThaw(char *block) {
    return SemiThaw<T>(block);
  }

  /** Behavior similar to SemiCopy; beware memory leaks */
  template<typename T>
  COMPILER_DEPRECATED
  void PointerRelocate(const char *old_loc, char *new_loc) {
    /*
     * TODO: Perhaps detect if Shallow, doing nothing, or try function
     * FixTransients, which by default deallocates them and
     * reallocates them and can be redefined to blank.  Carefully
     * consider Swap behavior in cachearray_impl.h.
     */
    // leaks memory for allocated transients
    SemiCopy<T>(new_loc, old_loc);
  }
};

////////// Deprecated ////////////////////////////////////////////////

#define OT_MY_OBJECT OT_OBJ
#define OT_MY_ARRAY OT_STATIC_ARRAY
#define OT_MALLOC_ARRAY OT_ALLOC
#define OT_PTR_NULLABLE OT_PTR

#define OT_FIX OT_REFILL_TRANSIENTS

#define OT_DEF OBJECT_TRAVERSAL
#define OT_DEF_BASIC OBJECT_TRAVERSAL

#endif /* BASE_OTRAV_H */
