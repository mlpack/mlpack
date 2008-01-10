/**
 * @file otrav.h
 *
 * Object-tree traversal.
 *
 * See the comment on namespace @c ot;
 */

#ifndef BASE_OTRAV_H
#define BASE_OTRAV_H

#include "ccmem.h"

#include <typeinfo>
#include <stdarg.h>
#include <ctype.h>

// Declares the name of the field, for printing
#define OT__NAME_(x) (v_OT->Name( #x ))

/**
 * Within OT_DEF, declare a sub-object (or primitive) that is directly
 * contained, NOT pointed to.
 *
 * (For those reading the definition: v_OT is the parameter, the visitor,
 * passed to the object traverse member function).
 */
#define OT_MY_OBJECT(x) (OT__NAME_(x), v_OT->MyObject(this->x))
/**
 * Within OT_DEF, declare a static-sized array embedded within your object.
 *
 * The length of the array is determined automatically via sizeof.
 */
#define OT_MY_ARRAY(x) (OT__NAME_(x), v_OT->MyArray(this->x, sizeof(this->x) / sizeof(this->x[0])))
/**
 * Within OT_DEF, declare an object being pointed to, managed by
 * new and delete.
 */
#define OT_PTR(x) (OT__NAME_(x), v_OT->Ptr(this->x, false))
/**
 * Within OT_DEF, declare an array being pointed to, managed by
 * new[] and delete[].
 */
#define OT_ARRAY(x, i) (OT__NAME_(x), v_OT->Array(this->x, i))
/**
 * Within OT_DEF, declare an array or object being pointed to managed by
 * malloc and free.
 */
#define OT_MALLOC_ARRAY(x, i) (OT__NAME_(x), v_OT->MallocArray(this->x, i))
/**
 * Within OT_DEF, declare a pointer to a new-delete object that might be NULL.
 */
#define OT_PTR_NULLABLE(x) (OT__NAME_(x), v_OT->Ptr(this->x, true))

/**
 * Like OT_DEF but doesn't define any of the automatic freebies.
 *
 * @see OT_DEF
 */
#define OT_DEF_ONLY(AClass) \
 public: \
  template<typename Visitor> \
  friend void TraverseObject(AClass *obj_OT, Visitor *v_OT) { \
    obj_OT->TraverseObject__OT_(v_OT); \
  } \
 private: \
  template<typename Visitor> \
  void TraverseObject__OT_(Visitor *v_OT)

/**
 * Automatically generate a copy constructor based on the object traversal.
 *
 * Note that OT_DEF calls this automatically.
 */
#define OT_GEN_COPY_CONSTRUCTOR(AClass) \
 public: AClass(const AClass& other) { \
   ot__private::ZOTDeepCopier::Doit(other, this); \
 } \
 private:

/**
 * Automatically create a dstructor based on object traversal.
 *
 * Note that OT_DEF calls this automatically.
 */
#define OT_GEN_DESTRUCTOR(AClass) \
 public: ~AClass() { ot__private::DestructorImplementation(this); } private:

/**
 * Automatically create a dstructor based on object traversal.
 *
 * Note that OT_DEF calls this automatically.
 */
#define OT_GEN_ASSIGN(AClass) \
 public: const AClass& operator = (const AClass& other) \
     { this->~AClass(); new(this)AClass(other); return *this; } \
 private:

/**
 * Generate a default constructor.
 *
 * Note that OT_DEF calls this automatically.
 * TODO(gboyer): It would be useful if this somehow poisoned the memory.
 */
#define OT_GEN_DEFAULT_CONSTRUCTOR(AClass) \
 public: AClass() { } private:

/**
 * Automatically create a Copy method based on the copy constructor.
 *
 * Note that OT_DEF calls this automatically.
 */
#define OT_GEN_COPY_METHOD(AClass) \
 public: void Copy(const AClass& other) { new(this)AClass(other); } private:

/**
 * Define the object traversal for this object, and clearly defines its
 * lifecycle and resource allocation as FASTlib-compliant.
 *
 * Example:
 * @code
 * class MyTree {
 *  private:
 *   int value;
 *   MyTree *left;
 *   MyTree *right;
 *   int num_extra_data;
 *   Data *extra_data_array;
 *
 *   OT_DEF(MyTree) {
 *     OT_MY_OBJECT(value);
 *     OT_PTR_NULLABLE(left);
 *     OT_PTR_NULLABLE(right);
 *     OT_MY_OBJECT(num_extra_data);
 *     OT_ARRAY(extra_data_array, num_extra_data);
 *   }
 * };
 * ... rest of class definition ...
 * @endcode
 *
 * The OT_DEF is used to declare a class's members, and its pointers.
 * Notice that <code>OT_MY_OBJECT(num_extra_data)</code> must come before
 * the subsequent line that uses num_extra_data as an array length.  If
 * deserialization is occuring, each <code>OT_...</code> call is actually
 * deserializing each member, so num_extra_data is uninitialized until
 * <code>OT_MY_OBJECT</code> is called on it.
 *
 * Fine-point: If you have an array of pointers, you are pretty much doomed
 * to declare the array of pointers and iterate over the array yourself for
 * each pointer, treating each element of the array as a separate pointer.
 * Note that all your familiar programming concepts like if-statements and
 * for-loops are valid in the OT_DEF block because it's just a function,
 * just make sure you're careful about only accessing members that have
 * already been "traversed".
 *
 * This macro will also make your class FASTlib-compliant by creating a
 * default constructor, copy constructor, assignment operator,
 * and Copy method.
 *
 * @see OT_MY_OBJECT, OT_MY_ARRAY, OT_PTR, OT_ARRAY, OT_MALLOC_ARRAY,
 * OT_PTR_NULLABLE, OT_ARRAY_NULLABLE, OT_MALLOC_ARRAY_NULLABLE.
 */
#define OT_DEF(AClass) \
 public: \
  OT_GEN_DEFAULT_CONSTRUCTOR(AClass) \
  OT_GEN_COPY_CONSTRUCTOR(AClass) \
  OT_GEN_DESTRUCTOR(AClass) \
  OT_GEN_COPY_METHOD(AClass) \
  OT_GEN_ASSIGN(AClass) \
  OT_DEF_ONLY(AClass)

/**
 * Defines object traversal for classes with no pointer members.
 *
 * This will use the compiler's default copy constructor and default
 * destructor, which are almost certainly faster than the ot-based one.
 */
#define OT_DEF_BASIC(AClass) \
 public: \
  OT_GEN_DEFAULT_CONSTRUCTOR(AClass) \
  OT_GEN_COPY_METHOD(AClass) \
  OT_GEN_ASSIGN(AClass) \
  OT_DEF_ONLY(AClass)

/**
 * Declares automatic OT features for a class for classes that have no
 * pointers.
 *
 * This ensures you have a valid,
 */
#define OT_DEFAULTS

/**
 * Declare automatic OT features for a class, but for classes that have
 * pointers.
 */
#define OT_DEFAULTS_PTR

// Re-think how this is supposed to work.
// /**
//  * Create an automatically-generated print method for your class.
//  */
// #define OT_GENERATE_PRINT(AClass)
//  public:
//   template<>
//   friend void Print(const AClass& obj, FILE *stream) {
//     OTPrint(obj, stream);
//   }

// TODO: Automatically generate copy constructors and the like

/**
 * Like OT_DEF, but automatically generates as many standard methods as
 * possible.
 */
#define OT_FULL(AClass) \
  OT_GENERATE_PRINT(AClass) \
  OT_DEF(AClass)

/**
 * Specify a clean-up step to run after deserialization, for instance, to
 * populate transient fields.
 *
 * An example is ArrayList - it has both a length and capacity.  The capacity
 * need not be stored, but upon deserialization, the capacity must be
 * initialized to a valid value, such as the length.
 *
 * This is only called when the object has valid pointers.  However, this
 * must not allocate any resources, because OT_FIX on the class may be called
 * multiple times.
 */
#define OT_FIX(AClass) \
 public: \
  friend void TraverseObjectPostprocess(AClass *x) { \
    x->TraverseObjectPostprocess__OT_(); \
  } \
 private: \
  void TraverseObjectPostprocess__OT_()


// The object-tree-visitor interface.
// class OTBlankVisitor {
//  public:
//   /** visits an object with no OT implementation */
//   template<typename T> void Primitive(T& x);
//   /** visits an internal object */
//   template<typename T> void MyObject(T& x);
//   /** visits an array */
//   template<typename T> void MyArray(T* x, index_t i);
//   /** visits an object pointed to, allocated with new */
//   template<typename T> void Ptr(T*& x, bool nullable);
//   /** visits an array pointed to, allocated with new[] */
//   template<typename T> void Array(T*& x, index_t i, bool nullable);
//   /** visits an array pointed to, allocated with malloc */
//   template<typename T> void MallocArray(T*& x, index_t i, bool nullable);
// };

/**
 * Perform object-tree traversal on a single object with a given object-tree
 * visitor.
 *
 * The visitor can perform pretty much any function it wants with the
 * contents of each data type.  It can print, serialize, deserialize,
 * pointer-freeze, etc.
 */
template<typename T, typename Visitor>
inline void TraverseObject(T* x, Visitor* v) {
  v->Primitive(*x);
}

/**
 * Postprocess function for making copies, to fix anything that may be
 * inaccurate from a plain copy.
 *
 * You will probably never need to implement this -- if you do, use
 * OT_FIX or see its comments.
 */
template<typename T>
inline void TraverseObjectPostprocess(T* x) {
}

/**
 * Traverses an array with a particular visitor.
 *
 * This is a convenience method that just calls TraverseObject on each
 * element.
 */
template<typename T, typename Visitor>
inline void TraverseArray(T* x, index_t n_elems, Visitor *v) {
  for (index_t i = 0; i < n_elems; i++) {
    TraverseObject(&x[i], v);
  }
}

#include "otrav_impl.h"

/**
 * Object-traversal utilities, such as serialization and deserialization.
 *
 * This is for traversing a directed acyclic graph of pointers, i.e. the
 * actual underlying data structure.  It turns out a generalized DAG
 * traversal framework allows for the following to be available at no
 * additional work on the application programmer:
 *
 * @li Serialization (save to byte stream)
 * @li Deserialization (read from byte stream)
 * @li Object freezing/thawing/refreezing
 *     (storing bulk flattened objects in RAM)
 * @li Debug print, or save to s-expression or XML
 * @li Destructors and copy constructors
 *
 * We define the concept of an object-traversal (OT) compliant class:
 *
 * @li No polymorphism, i.e. virtual functions
 * @li No cycles in the pointer graph (actually there are a few exceptions,
 *     such as trees can have parent pointers, see OT_FIX)
 * @li No inheritance, although non-polymorphic inheritance may be a
 *     possibility
 * @li Blank default constructor that puts object into an "invalid" state,
 *     with Init methods
 *
 * These classes can be thought of as "pure data structures".  If your
 * class is not OT-compliant, you probably want to put the
 * FORBID_ACCIDENTAL_COPIES header in your class.  This is not to say
 * that non-OT-compliant classes aren't useful -- it's definitely true
 * that some objects aren't really meant to be copied or sent over the
 * network.  Think for example the Thread class.
 *
 * To define the object traversal for a class, use @c OT_DEF (which generates
 * function headers) and fill out the function with
 * @c OT_MY_OBJECT(member_field) and other relevant macros within the
 * body.  See THOR for many simple examples, and see col/arraylist.h
 * for a rather complicated example.
 */
namespace ot {

  /**
   * Prints any OT-defined object.
   *
   * Perfect for debugging.
   */
  template<typename T>
  void Print(const T& object, FILE *stream = stderr) {
    ot__private::ZOTPrinter printer;
    printer.Doit(object, stream);
  }

  /**
   * Prints any object, for use in print statements.
   *
   * What this does is return the message you provide it, so that it is
   * reasonable to put this in the variable-arguments list of a printf
   * messge:
   *
   * @code
   * DEBUG_ASSERT_MSG(!cat.is_wet(), "Cat is wet (see below %s)!",
   *    ot::PrintMsg(cat, "cat"));
   * @endcode
   */
  template<typename T>
  const char *PrintMsg(const T& object, const char *message) {
    ot__private::ZOTPrinter printer;
    fprintf(stderr, ANSI_HRED"---- PRINTING %s ----"ANSI_CLEAR"\n", message);
    printer.Doit(object, stderr);
    return message;
  }

  /**
   * Finds the number of bytes required to freeze an object.
   */
  template<typename T>
  size_t PointerFrozenSize(const T& obj) {
    ot__private::ZOTFrozenSizeCalculator calc;
    calc.Doit(obj);
    return calc.size();
  }

  /**
   * Makes a copy of an object, freezing it for the first time.
   */
  template<typename T>
  void PointerFreeze(const T& live_object, char *block) {
    ot__private::ZOTPointerFreezer freezer;
    freezer.Doit(live_object, block);
    DEBUG_SAME_INT(freezer.size(), ot::PointerFrozenSize(live_object));
  }

  /**
   * Takes an object that is laid out serially, and adjusts all its pointers
   * so that they are normalized to zero.
   */
  template<typename T>
  void PointerRefreeze(T* obj) {
    ot__private::ZOTPointerRelocator fixer;
    fixer.Doit<T>(
        0, -mem::PtrAbsAddr(obj),
        reinterpret_cast<T*>(obj));
  }

  /**
   * Takes an object that is laid out serially, and adjusts all its pointers
   * so that they are normalized to zero.
   *
   * This assumes that "dest" is an object that is laid out serially, but
   * all its pointers are as if it had been copied from src.  This is used
   * for reading from an existing cache -- the pointers are fixed in a
   * temporary buffer rather than in the cache, so that other threads do not
   * experience any negative side effects.
   */
  template<typename T>
  void PointerRefreeze(const T* src, char* dest) {
    ot__private::ZOTPointerRelocator fixer;
    fixer.Doit<T>(
        mem::PtrDiffBytes(dest, src), -mem::PtrAbsAddr(src),
        reinterpret_cast<T*>(dest));
  }

  /**
   * Takes an object that is laid out serially with all its pointers
   * normalized to zero, and makes all the pointers live again.
   *
   * Note that this is the function which calls the postprocess function
   * (OT_FIX)!
   */
  template<typename T>
  T* PointerThaw(char *block) {
    ot__private::ZOTPointerThawer fixer;
    return fixer.Doit<T>(
        mem::PtrAbsAddr(block),
        block);
  }

  /**
   * Relocates an object from a previous location to a new location.
   *
   * Call this to fix pointers after swapping or memcopying an object.
   */
  template<typename T>
  void PointerRelocate(const char *old_location, char *new_location) {
    ot__private::ZOTPointerRelocator fixer;
    fixer.Doit<T>(
        mem::PtrDiffBytes(new_location, old_location),
        mem::PtrDiffBytes(new_location, old_location),
        reinterpret_cast<T*>(new_location));
  }

  /**
   * Deep-copy initializer for OT-compliant classes.
   */
  template<typename T>
  void Copy(const T& src, T* dest) {
    new(dest)T(src);
  }
};

#endif
