/**
 * @file otrav.h
 *
 * Object-tree traversal.
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
 * These classes can be thought of as "pure data structures".  If your class
 * is not OT-compliant, you probably want to put the FORBID_COPY header in
 * your class.  This is not to say that non-OT-compliant classes aren't
 * useful -- it's definitely true that some objects aren't really meant to
 * be copied or sent over the network.  Think for example the Thread class.
 */

#ifndef BASE_OTRAV_H
#define BASE_OTRAV_H

#include "ccmem.h"

#include <typeinfo>
#include <stdarg.h>
#include <ctype.h>

#define OT__NAME(x) v_OT->Name( #x )

/**
 * Within OT_DEF, declare a sub-object (or primitive) that is directly
 * contained, NOT pointed to.
 */
#define OT_MY_OBJECT(x) (OT__NAME(x), v_OT->MyObject(this->x))
/**
 * Within OT_DEF, declare a static-sized array embedded within your object.
 *
 * The length of the array is determined automatically via sizeof.
 */
#define OT_MY_ARRAY(x) (OT__NAME(x), v_OT->MyArray(this->x, sizeof(this->x) / sizeof(this->x[0])))
/**
 * Within OT_DEF, declare an object being pointed to, managed by
 * new and delete.
 */
#define OT_PTR(x) (OT__NAME(x), v_OT->Ptr(this->x, false))
/**
 * Within OT_DEF, declare an array being pointed to, managed by
 * new[] and delete[].
 */
#define OT_ARRAY(x, i) (OT__NAME(x), v_OT->Array(this->x, i))
/**
 * Within OT_DEF, declare an array or object being pointed to managed by
 * malloc and free.
 */
#define OT_MALLOC_ARRAY(x, i) (OT__NAME(x), v_OT->MallocArray(this->x, i))
/**
 * Within OT_DEF, declare a pointer to a new-delete object that might be NULL.
 */
#define OT_PTR_NULLABLE(x) (OT__NAME(x), v_OT->Ptr(this->x, true))

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
 */
#define OT_GEN_COPY_CONSTRUCTOR(AClass) \
 public: AClass(const AClass& other) { ot_private::DeepCopyImplementation(other, this); } private:

/**
 * Automatically create a dstructor based on object traversal.
 */
#define OT_GEN_DESTRUCTOR(AClass) \
 public: ~AClass() { ot_private::DestructorImplementation(this); } private:

/**
 * Automatically create a dstructor based on object traversal.
 */
#define OT_GEN_ASSIGN(AClass) \
 public: const AClass& operator = (const AClass& other) \
     { this->~AClass(); new(this)AClass(other); return *this; } \
 private:

/**
 * Generate a default constructor.
 */
#define OT_GEN_DEFAULT_CONSTRUCTOR(AClass) \
 public: AClass() { } private:

/**
 * Automatically create a Copy method based on the copy constructor.
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
 * You will probably never need to implement this.  This exists
 * mainly so that lazy-rezing data structures (i.e. ArrayList) can serialize
 * themselves as their trimmed size -- the TraverseObject function neglects
 * saving the capacity, and fills in the capacity upon deserialization.
 * Note this should NOT dereference any pointers within the object, just
 * update things like flags.
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

/**
 * Private namespace for object-traversal utilities.
 */
namespace ot_private {
  // TODO: Space-conservatory serialization and deserialization
  // (Currently only freezing/thawing is supported)

  template<typename DefaultPrinter, typename Printer, typename T>
  struct OTPrinter_Dispatcher {
    static void Print(const char *name, T& x, Printer* printer) {
      DefaultPrinter::Print(name, x, printer);
    }
  };

  /* macro for use within this file */
  #define OTPRINTER__SPECIAL(T, format_str) \
    template<typename DefaultPrinter, typename Printer> \
    struct OTPrinter_Dispatcher<DefaultPrinter, Printer, T> { \
      static void Print(const char *name, T x, Printer *printer) { \
        printer->Write("%s : "format_str, name, x); \
      } \
    };
  OTPRINTER__SPECIAL(const char*, "string = %s");
  OTPRINTER__SPECIAL(char, "char = %d");
  OTPRINTER__SPECIAL(short, "short = %d");
  OTPRINTER__SPECIAL(int, "int = %d");
  OTPRINTER__SPECIAL(long, "long = %ld");
  OTPRINTER__SPECIAL(unsigned char, "char = %u");
  OTPRINTER__SPECIAL(unsigned short, "short = %u");
  OTPRINTER__SPECIAL(unsigned int, "int = %u");
  OTPRINTER__SPECIAL(unsigned long, "long = %lu");
  OTPRINTER__SPECIAL(float, "float = %f");
  OTPRINTER__SPECIAL(double, "double = %f");

  /**
   * Utility class to take an OT-compatible object and prints it to screen.
   */
  class OTPrinter {
   private:
    FILE *stream_;
    int indent_amount_;
    const char *name_;

   private:
    template<typename T>
    struct DefaultPrimitivePrinter {
      static void Print(const char *name, const T& x, OTPrinter *printer) {
        printer->ShowIndents();
        for (size_t i = 0; i < sizeof(T); i++) {
          fprintf(printer->stream(), " %02X",
              reinterpret_cast<const unsigned char*>(&x)[i]);
        }
        fprintf(printer->stream(), "\n");
      }
    };

    template<typename T>
    struct DefaultObjectPrinter {
      static void Print(const char *name, T& x, OTPrinter *printer) {
        printer->Write("%s : %s {", name, typeid(T).name());
        printer->Indent(2);
        TraverseObject(&x, printer);
        printer->Indent(-2);
        printer->Write("}");
      }
    };

   public:
    template<typename T>
    void Doit(const T& x, FILE *stream_in) {
      stream_ = stream_in;
      indent_amount_ = 0;
      name_ = "<root>";
      Object(const_cast<T*>(&x), false, "");
    }

    /** Stores the name of the object going to come in. */
    void Name(const char *s) {
      name_ = s;
    }

    template<typename T> void Primitive(T& x) {
      OTPrinter_Dispatcher< DefaultPrimitivePrinter<T>, OTPrinter, T >
          ::Print(name_, x, this);
    }

    template<typename T> void Object(T* obj, bool nullable,
        const char *label) {
      if (nullable && !obj) {
        Write("%s : %s %s = NULL", name_, label, typeid(T).name());
      } else {
        OTPrinter_Dispatcher< DefaultObjectPrinter<T>, OTPrinter, T >
            ::Print(name_, *obj, this);
      }
    }

    template<typename T> void Array(T* array, index_t len) {
      if (array == NULL) {
        len = 0;
      }
      Write("%s : %s[%"LI"d] = {", name_, typeid(T).name(), len);
      Indent(2);
      for (index_t i = 0; i < len; i++) {
        Write("element %"LI"d {", i);
        Indent(2);
        name_ = "(array element)";
        TraverseObject(&array[i], this);
        Indent(-2);
        Write("}");
      }
      Indent(-2);
      Write("}");
    }

    /** Visits an internal object. */
    template<typename T> void MyObject(T& x) {
      // Recurse in case this sub-object has pointers
      Object(&x, false, "embedded");
    }
    /** Visits an array. */
    template<typename T> void MyArray(T* x, index_t len) {
      // Recurse in case any of these objects have pointers
      Array(x, len);
    }

    /**
     * Visits an object pointed to, allocated with new.
     *
     * This allocates space within the block for the pointer, copies the
     * data pointed to, and recurses on the data pointed to.
     */
    template<typename T> void Ptr(T*& source_region, bool nullable) {
      Object(source_region, nullable, "pointer-to");
    }

    /** Visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& source_region, index_t len) {
      Array(source_region, len);
    }

   public:
    void Indent(int delta) {
      indent_amount_ += delta;
    }

    void Write(const char *format, ...);

    void ShowIndents();

    FILE *stream() const {
      return stream_;
    }
  };

  /**
   * Takes an OT-compatible object and saves a linear copy in a block of
   * memory.
   *
   * This is analogous to serialization but distinct.  Serialization does
   * not allocate space for transient fields such as pointers.  However, this
   * dumps every object in its entirety, with the hope that bringing the
   * object "back to life" is very quick.  When stored, each pointer is
   * normalized to zero, and the object can be brought back to life by just
   * renormalizing all the pointers.
   *
   * The code here is far more complex than I expected it to be -- please
   * read the comments!
   *
   * ANY MODIFICATIONS TO THIS MUST ALSO BE MADE TO THE SIZE CALCULATOR!
   */
  class OTPointerFreezer {
   private:
    /** The block of memory to freeze into. */
    char *block_;
    /** The current position within the block. */
    ptrdiff_t pos_;
    /**
     * For updating pointers with normalized pointers, this is the difference
     * between the destination and source regions for the *current* object
     * being considered.
     */
    ptrdiff_t freeze_offset_;

   public:
    template<typename T>
    void Doit(const T& x, char *block_in) {
      block_ = block_in;
      pos_ = sizeof(T);
      freeze_offset_ = mem::PointerDiff(block_, &x);

      mem::Copy(reinterpret_cast<T*>(block_), &x);
      // we must cast away const due to TraverseObject's limitations
      TraverseObject(const_cast<T*>(&x), this);
    }

    size_t size() const {
      return stride_align_max(pos_);
    }

    /** Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /** Visits an object with no OT implementation. */
    template<typename T> void Primitive(T& x) {
      // Primitives can be bit-copied
    }

    /** Visits an internal object. */
    template<typename T> void MyObject(T& x) {
      // Recurse in case this sub-object has pointers
      TraverseObject(&x, this);
    }
    /** Visits an array. */
    template<typename T> void MyArray(T* x, index_t len) {
      // Recurse in case any of these objects have pointers
      TraverseArray(x, len, this);
    }

    /**
     * Visits an object pointed to, allocated with new.
     *
     * This allocates space within the block for the pointer, copies the
     * data pointed to, and recurses on the data pointed to.
     */
    template<typename T> void Ptr(T*& source_region, bool nullable);

    /**
     * Visits an array pointed to, allocated with new[].
     *
     * This allocates space within the block for the array, copies the
     * data pointed to, and recurses on the array's elements.
     */
    template<typename T> void Array(T*& source_region, index_t len);

    /** Visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& source_region, index_t len) {
      Array(source_region, len);
    }

   private:
    template <typename T>
    /**
     * Gets a pointer to the pointer in the destination region that needs
     * to be updated.  A picture might help.
     *
     * @param source_region_ptr the pointer to the original pointer, in
     *        its original location within the larger structure, used with
     *        pointer arithmetic for updating the resulting pointers
     */
    T** DestinationEquivalentPointer_(T** source_region_ptr) {
      return mem::PointerAdd(source_region_ptr, freeze_offset_);
    }
    /**
     * Aligns the current position to the given stride, and returns a
     * normalized-to-zero pointer for its data, fixing the result pointer
     * too.
     *
     * In reality, this is just a couple assembly instructions.
     *
     * @param source_region_ptr the pointer to the original pointer, in
     *        its original location within the larger structure, used with
     *        pointer arithmetic for updating the resulting pointers
     */
    template <typename T>
    T* TranslateAndFixPointer_(T** source_region_ptr) {
      // Make sure we are aligned to the proper alignment for the data
      pos_ = stride_align(pos_, T);
      // Find the pointer in the frozen block by adding the "freeze offset"
      // This offset basically says "Given some memory within the live object
      // that is being frozen, find the corresponding memory within the
      // object that is being frozen".
      T** pointer_to_fix = DestinationEquivalentPointer_(source_region_ptr);
      // We already copied the source region to the destination we are
      // considering, so the value of these two pointers should be equal.
      DEBUG_ASSERT_MSG(*pointer_to_fix == *source_region_ptr,
          "%p != %p", *pointer_to_fix, *source_region_ptr);
      // Now, we normalize the pointer such that zero is the beginning of the
      // chynk of memory.
      *pointer_to_fix = reinterpret_cast<T*>(pos_);
      // Return the pointer within the block where future accesses should occur.
      return reinterpret_cast<T*>(block_ + pos_);
    }
  };

  template<typename T> void OTPointerFreezer::Ptr(
      T*& source_region, bool nullable) {
    if (nullable && unlikely(source_region == NULL)) {
      *DestinationEquivalentPointer_(&source_region) = NULL;
    } else {
      // Get the pointer we will write into, and fix our internal pointer
      T* dest = TranslateAndFixPointer_(&source_region);
      // Copy the object and progress
      pos_ += sizeof(T);
      mem::Copy(dest, source_region);
      // Save our old freeze offset
      size_t freeze_offset_tmp = freeze_offset_;
      // Calculate new freeze offset as the distance between the source and
      // destination memory regions.
      freeze_offset_ = mem::PointerDiff(dest, source_region);
      // Recurse on the object.
      TraverseObject(source_region, this);
      // Revert to the old freeze offset.
      freeze_offset_ = freeze_offset_tmp;
    }
  }

  template<typename T> void OTPointerFreezer::Array(
      T*& source_region, index_t len) {
    if (len == 0) {
      *DestinationEquivalentPointer_(&source_region) = NULL;
    } else {
      // Get the pointer we will write into, and fix our internal pointer
      T* dest = TranslateAndFixPointer_(&source_region);
      // Calculate the total size allocated, copy, and progress
      size_t size = len * sizeof(T);
      pos_ += size;
      mem::CopyBytes(dest, source_region, size);
      // Save old freeze offset
      size_t freeze_offset_tmp = freeze_offset_;
      // Calculate new freeze offset
      freeze_offset_ = mem::PointerDiff(dest, source_region);
      // Recurse over each object
      for (index_t i = 0; i < len; i++) {
        TraverseObject(&source_region[i], this);
      }
      // Restore old freeze offset because we have returned to the old object
      freeze_offset_ = freeze_offset_tmp;
    }
  }

  class OTFrozenSizeCalculator {
   private:
    size_t pos_;

   public:
    template<typename T>
    void Doit(const T& obj) {
      pos_ = sizeof(T);
      TraverseObject(const_cast<T*>(&obj), this);
    }

    /**
     * Returns the calculated size.
     */
    size_t size() const {
      return stride_align_max(pos_);
    }

    /** Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /** visits an object with no OT implementation */
    template<typename T> void Primitive(T& x) {}
    /** visits an internal object */
    template<typename T> void MyObject(T& x) {
      TraverseObject(&x, this);
    }
    /** visits an array */
    template<typename T> void MyArray(T* x, index_t len) {
      TraverseArray(x, len, this);
    }

    /** visits an object pointed to, allocated with new */
    template<typename T> void Ptr(T*& x, bool nullable) {
      if (!nullable || x != NULL) {
        PretendLayout_<T>(1);
        TraverseObject(x, this);
      }
    }
    /** visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len) {
      if (len != 0) {
        PretendLayout_<T>(len);
        TraverseArray(x, len, this);
      }
    }
    /** visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len) {
      Array(x, len);
    }

   private:
    template<typename T>
    void PretendLayout_(index_t count) {
      pos_ = (stride_align(pos_, T)) + (sizeof(T) * count);
    }
  };

  class OTPointerThawer {
   private:
    ptrdiff_t offset_;

   public:
    template<typename T>
    T* Doit(ptrdiff_t offset_in, char *data) {
      offset_ = offset_in;
      T* dest = reinterpret_cast<T*>(data);
      MyObject(*dest);
      return dest;
    }

    template<typename T>
    T* Doit(char *data) {
      return Doit<T>(data, reinterpret_cast<ptrdiff_t>(data));
    }

    /** Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /** visits an object with no OT implementation */
    template<typename T> void Primitive(T& x) {}
    /** visits an internal object */
    template<typename T> void MyObject(T& x) {
      TraverseObject(&x, this);
      TraverseObjectPostprocess(&x);
    }
    /** visits an array */
    template<typename T> void MyArray(T* x, index_t len) {
      for (index_t i = 0; i < len; i++) {
        MyObject(x[i]);
      }
    }
    /** visits an object pointed to, allocated with new */
    template<typename T> void Ptr(T*& x, bool nullable) {
      if (!nullable || x != NULL) {
        x = mem::PointerAdd(x, offset_);
        MyObject(*x);
      }
    }
    /** visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len) {
      if (len != 0) {
        x = mem::PointerAdd(x, offset_);
        MyArray(x, len);
      }
    }
    /** visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len) {
      Array(x, len);
    }
  };

  class OTPointerRelocator {
   private:
    ptrdiff_t pre_offset_;
    ptrdiff_t post_offset_;

   public:
    /**
     * Fixes pointers.
     *
     * @param pre_offset_in the offset between where the pointers are
     *        currently pointing, and where they would need to point in
     *        order to recurse on the data structure (no modifications made)
     * @param post_offset_in the offset between where the pointers are
     *        currently pointing, and the new address space they are relocated
     *        to
     * @param dest the object to recurse on
     */
    template<typename T>
    T* Doit(ptrdiff_t pre_offset_in, ptrdiff_t post_offset_in, T *dest) {
      pre_offset_ = pre_offset_in;
      post_offset_ = post_offset_in;
      TraverseObject(dest, this);
      return dest;
    }

    /** Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /** visits an object with no OT implementation */
    template<typename T> void Primitive(T& x) {}
    /** visits an internal object */
    template<typename T> void MyObject(T& x) {
      TraverseObject(&x, this);
    }
    /** visits an array */
    template<typename T> void MyArray(T* x, index_t len) {
      TraverseArray(x, len, this);
    }
    /** visits an object pointed to, allocated with new */
    template<typename T> void Ptr(T*& x, bool nullable) {
      if (!nullable || x != NULL) {
        TraverseObject(mem::PointerAdd(x, pre_offset_), this);
        x = mem::PointerAdd(x, post_offset_);
      }
    }
    /** visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len) {
      if (len != 0) {
        TraverseArray(mem::PointerAdd(x, pre_offset_), len, this);
        x = mem::PointerAdd(x, post_offset_);
      }
    }
    /** visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len) {
      Array(x, len);
    }
  };

  struct OTDeepCopier {
   public:
    /** Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /** visits an object with no OT implementation */
    template<typename T> void Primitive(T& x) {}
    /** visits an internal object */
    template<typename T> void MyObject(T& x) {
      TraverseObject(&x, this);
    }
    /** visits an array */
    template<typename T> void MyArray(T* x, index_t len) {
      TraverseArray(x, len, this);
    }
    /** visits an object pointed to, allocated with new */
    template<typename T> void Ptr(T*& x, bool nullable) {
      if (!nullable || x != NULL) {
        x = new T(*x);
      }
    }
    /** visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len) {
      x = mem::CopyConstruct(new T[len], x, len);
    }
    /** visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len) {
      x = mem::CopyConstruct(mem::Alloc<T>(len), x, len);
    }
  };

  struct OTDestructor {
   public:
    /** Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /** visits an object with no OT implementation */
    template<typename T> void Primitive(T& x) {}
    /** visits an internal object */
    template<typename T> void MyObject(T& x) {
      // C++ will automatically chain this
    }
    /** visits an array */
    template<typename T> void MyArray(T* x, index_t len) {
      // C++ will automatically chain this
    }
    /** visits an object pointed to, allocated with new */
    template<typename T> void Ptr(T*& x, bool nullable) {
      if (!nullable || x != NULL) {
        delete x;
      }
      DEBUG_POISON_PTR(x);
    }
    /** visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len, bool nullable) {
      delete[] x;
      DEBUG_POISON_PTR(x);
    }
    /** visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len) {
      T *tmpx = x;
      mem::DestructAll(tmpx, len);
      mem::Free(tmpx);
      DEBUG_POISON_PTR(x);
    }
  };

  template<typename T>
  void DestructorImplementation(T *dest) {
    ot_private::OTDestructor d;
    TraverseObject(dest, &d);
    // can't poison this because of destructor chanining
  }

  template<typename T>
  void DeepCopyImplementation(const T& src, T *dest) {
    ot_private::OTDeepCopier d;
    mem::Copy(dest, &src, 1);
    TraverseObject(dest, &d);
  }
}; // namespace ot_private

namespace ot {

  /**
   * Prints any object.
   *
   * Perfect for debugging.
   */
  template<typename T>
  void Print(const T& object, FILE *stream = stderr) {
    ot_private::OTPrinter printer;
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
    ot_private::OTPrinter printer;
    fprintf(stderr, ANSI_HRED"---- PRINTING %s ----"ANSI_CLEAR"\n", message);
    printer.Doit(object, stderr);
    return message;
  }

  /**
   * Finds the number of bytes required to freeze an object.
   */
  template<typename T>
  size_t PointerFrozenSize(const T& obj) {
    ot_private::OTFrozenSizeCalculator calc;
    calc.Doit(obj);
    return calc.size();
  }

  /**
   * Makes a copy of an object, freezing it for the first time.
   */
  template<typename T>
  void PointerFreeze(const T& live_object, char *block) {
    ot_private::OTPointerFreezer freezer;
    freezer.Doit(live_object, block);
    DEBUG_SAME_INT(freezer.size(), ot::PointerFrozenSize(live_object));
  }

  /**
   * Takes an object that is laid out serially, and adjusts all its pointers
   * so that they are normalized to zero.
   */
  template<typename T>
  void PointerRefreeze(T* obj) {
    ot_private::OTPointerRelocator fixer;
    fixer.Doit<T>(
        0, -mem::PointerAbsoluteAddress(obj),
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
    ot_private::OTPointerRelocator fixer;
    fixer.Doit<T>(
        mem::PointerDiff(dest, src), -mem::PointerAbsoluteAddress(src),
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
    ot_private::OTPointerThawer fixer;
    return fixer.Doit<T>(
        mem::PointerAbsoluteAddress(block),
        block);
  }

  /**
   * Relocates an object from a previous location to a new location.
   *
   * Call this to fix pointers after swapping or memcopying an object.
   */
  template<typename T>
  void PointerRelocate(const char *old_location, char *new_location) {
    ot_private::OTPointerRelocator fixer;
    fixer.Doit<T>(
        mem::PointerDiff(new_location, old_location),
        mem::PointerDiff(new_location, old_location),
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
