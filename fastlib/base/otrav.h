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
 * @li Serialization (save to disk)
 * @li Deserialization (read from disk)
 * @li Object freezing/thawing/refreezing
 *     (storing bulk flattened objects in RAM)
 * @li Debug print, or save to s-expression or XML
 * @li Destructors and copy constructors
 *
 *
 * This has no support for (at least currently):
 *
 * @li Cycles
 * @li Polymorphism (i.e. object-oriented inheritance)
 */


#ifndef BASE_OTRAV_H
#define BASE_OTRAV_H

#include "ccmem.h"

#include <typeinfo>
#include <stdarg.h>
#include <ctype.h>

// TODO: Remove nullability from arrays

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
#define OT_ARRAY(x, i) (OT__NAME(x), v_OT->Array(this->x, i, false))
/**
 * Within OT_DEF, declare an array or object being pointed to managed by
 * malloc and free.
 */
#define OT_MALLOC_ARRAY(x, i) (OT__NAME(x), v_OT->MallocArray(this->x, i, false))
/**
 * Within OT_DEF, declare a pointer to an object that might be NULL.
 */
#define OT_PTR_NULLABLE(x) (OT__NAME(x), v_OT->Ptr(this->x, true))
/**
 * Within OT_DEF, declare a pointer to an array that might be NULL.
 */
#define OT_ARRAY_NULLABLE(x, i) (OT__NAME(x), v_OT->Array(this->x, i, true))
/**
 * Within OT_DEF, declare a pointer to a malloced array that might be NULL.
 */
#define OT_MALLOC_ARRAY_NULLABLE(x, i) (OT__NAME(x), v_OT->MallocArray(this->x, i, true))

/**
 * Define the object traversal for this object.
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
 * The OT_DEF declares its own members, and its pointers.  Notice that
 * <code>OT_MY_OBJECT(num_extra_data)</code> must come before the subsequent
 * line that uses num_extra_data as an array length.  If deserialization is
 * occuring, each <code>OT_...</code> call is actually deserializing each
 * member, so num_extra_data is uninitialized until <code>OT_MY_OBJECT</code>
 * is called on it.
 *
 * Fine-point: If you have an array of pointers, you are pretty much doomed
 * to declare the array of pointers and iterate over the array yourself for
 * each pointer, treating each element of the array as a separate pointer.
 *
 * @see OT_MY_OBJECT, OT_MY_ARRAY, OT_PTR, OT_ARRAY, OT_MALLOC_ARRAY,
 * OT_PTR_NULLABLE, OT_ARRAY_NULLABLE, OT_MALLOC_ARRAY_NULLABLE.
 */
#define OT_DEF(AClass) \
 public: \
  template<typename Visitor> \
  friend void TraverseObject(AClass *obj_OT, Visitor *v_OT) { \
    obj_OT->TraverseObject__OT_(v_OT); \
  } \
 private: \
  template<typename Visitor> \
  void TraverseObject__OT_(Visitor *v_OT)

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
  
  // These have to be hoisted out of the class.
  // Apparently explicit specialization for templates cannot be done in class
  // scope.
  
  /** Visits an object with no OT implementation. */
  template<typename DefaultPrinter, typename Printer, typename T>
  void OTPrinter_Primitive(
      const char *name, T& x, Printer* printer) {
    DefaultPrinter::Print(name, x, printer);
  }
  template<typename DefaultPrinter, typename Printer>
  inline void OTPrinter_Primitive(
      const char *name, const char* x, Printer* printer) {
    printer->Write("%s : string = %s", name, x);
  }
  template<typename DefaultPrinter, typename Printer>
  inline void OTPrinter_Primitive(
      const char *name, char x, Printer* printer) {
    if (isprint(x)) {
      printer->Write("%s : char = %d '%c'", name, x, x);
    } else {
      printer->Write("%s : char = %d", name, x);
    }
  }
  template<typename DefaultPrinter, typename Printer>
  inline void OTPrinter_Primitive(
      const char *name, short x, Printer* printer) {
    printer->Write("%s : short = %d", name, x);
  }
  template<typename DefaultPrinter, typename Printer>
  inline void OTPrinter_Primitive(
      const char *name, int x, Printer* printer) {
    printer->Write("%s : int = %d", name, x);
  }
  template<typename DefaultPrinter, typename Printer>
  inline void OTPrinter_Primitive(
      const char *name, long x, Printer* printer) {
    printer->Write("%s : long = %ld", name, x);
  }
  template<typename DefaultPrinter, typename Printer>
  inline void OTPrinter_Primitive(
      const char *name, unsigned char x, Printer* printer) {
    printer->Write("%s : uchar = %u", name, x);
  }
  template<typename DefaultPrinter, typename Printer>
  inline void OTPrinter_Primitive(
      const char *name, unsigned short x, Printer* printer) {
    printer->Write("%s : ushort = %u", name, x);
  }
  template<typename DefaultPrinter, typename Printer>
  inline void OTPrinter_Primitive(
      const char *name, unsigned int x, Printer* printer) {
    printer->Write("%s : uint = %u", name, x);
  }
  template<typename DefaultPrinter, typename Printer>
  inline void OTPrinter_Primitive(
      const char *name, unsigned long x, Printer* printer) {
    printer->Write("%s : ulong = %lu", name, x);
  }
  template<typename DefaultPrinter, typename Printer>
  inline void OTPrinter_Primitive(
      const char *name, float x, Printer* printer) {
    printer->Write("%s : float = %f", name, x);
  }
  template<typename DefaultPrinter, typename Printer>
  inline void OTPrinter_Primitive(
      const char *name, double x, Printer* printer) {
    printer->Write("%s : double = %f", name, x);
  }

  /**
   * Takes an OT-compatible object and prints it to screen.
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
          fprintf(printer->stream(), " %02X", reinterpret_cast<const char*>(&x)[i]);
        }
        fprintf(printer->stream(), ")\n");
      }
    };

    template<typename T>
    struct DefaultObjectPrinter {
      static void Print(const char *name, T& x, OTPrinter *printer) {
        Write("%s : %s {", name, typeid(T).name());
        Indent(2);
        TraverseObject(&x, printer);
        Indent(-2);
        Write("}");
      }
    };

   public:
    template<typename T>
    void InitBegin(const T& x, FILE *stream_in) {
      stream_ = stream_in;
      indent_amount_ = 0;
      TraverseObject(const_cast<T*>(&x), this);
    }

    /** Stores the name of the object going to come in. */
    void Name(const char *s) {
      name_ = s;
    }

    template<typename T> void Primitive(const T& x) {
      OTPrinter_Primitive< DefaultPrimitivePrinter<T> >
          (name_, x, this);
    }

    template<typename T> void Object(T* obj, bool nullable,
        const char *label) {
      if (nullable && !obj) {
        Write("%s : %s %s = NULL", name_, label, typeid(T).name());
      } else {
        OTPrinter_Primitive< DefaultObjectPrinter<T> >
            (name_, *obj, this);
      }
    }

    template<typename T> void Array(T* array, index_t len,
        bool nullable) {
      if (nullable && !array) {
        Write("%s : %s[] = NULL", name_, typeid(T).name());
      } else {
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
    }

    /** Visits an internal object. */
    template<typename T> void MyObject(T& x) {
      // Recurse in case this sub-object has pointers
      Object(&x, false, "embedded");
    }
    /** Visits an array. */
    template<typename T> void MyArray(T* x, index_t len) {
      // Recurse in case any of these objects have pointers
      Array(x, len, false);
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
    template<typename T> void MallocArray(T*& source_region, index_t len,
        bool nullable) {
      Array(source_region, len, nullable);
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
    void InitBegin(const T& x, char *block_in) {
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
    template<typename T> void Array(T*& source_region, index_t len,
        bool nullable);

    /** Visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& source_region, index_t len,
        bool nullable) {
      Array(source_region, len, nullable);
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
      TraverseObjectPostprocess(dest);
      // Revert to the old freeze offset.
      freeze_offset_ = freeze_offset_tmp;
    }
  }

  template<typename T> void OTPointerFreezer::Array(
      T*& source_region, index_t len, bool nullable) {
    if (nullable && unlikely(source_region == NULL)) {
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
        TraverseObjectPostprocess(&dest[i]);
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
    void InitBegin(const T& obj) {
      pos_ = 0;
      PretendLayout_<T>(1);
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
    template<typename T> void Array(T*& x, index_t len, bool nullable) {
      if (!nullable || x != NULL) {
        PretendLayout_<T>(len);
        TraverseArray(x, len, this);
      }
    }
    /** visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len, bool nullable) {
      Array(x, len, nullable);
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
    T* InitBegin(char *data, ptrdiff_t offset_in) {
      offset_ = offset_in;
      T* dest = reinterpret_cast<T*>(data);
      TraverseObject(dest, this);
      return dest;
    }
    
    template<typename T>
    T* InitBegin(char *data) {
      InitBegin(data, reinterpret_cast<ptrdiff_t>(data));
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
        x = mem::PointerAdd(x, offset_);
        TraverseObject(x, this);
      }
    }
    /** visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len, bool nullable) {
      if (!nullable || x != NULL) {
        x = mem::PointerAdd(x, offset_);
        TraverseArray(x, len, this);
      }
    }
    /** visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len, bool nullable) {
      Array(x, len, nullable);
    }
  };

  class OTPointerRefreezer {
   private:
    ptrdiff_t offset_;
    
   public:
    template<typename T>
    T* InitBegin(T *dest) {
      offset_ = -reinterpret_cast<size_t>(dest);
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
        TraverseObject(x, this);
        x = mem::PointerAdd(x, offset_);
      }
    }
    /** visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len, bool nullable) {
      if (!nullable || x != NULL) {
        TraverseArray(x, len, this);
        x = mem::PointerAdd(x, offset_);
      }
    }
    /** visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len, bool nullable) {
      Array(x, len, nullable);
    }
  };
}; // namespace ot_private

namespace ot {

  template<typename T>
  void Print(const T& object, FILE *stream = stderr) {
    ot_private::OTPrinter printer;
    printer.InitBegin(object, stream);
  }

  /**
   * Finds the number of bytes required to freeze an object.
   */
  template<typename T>
  size_t PointerFrozenSize(const T& obj) {
    ot_private::OTFrozenSizeCalculator calc;
    calc.InitBegin(obj);
    return calc.size();
  }

  /**
   * Makes a copy of an object, freezing it for the first time.
   */
  template<typename T>
  void PointerFreeze(const T& live_object, char *block) {
    ot_private::OTPointerFreezer freezer;
    freezer.InitBegin(live_object, block);
    DEBUG_SAME_INT(freezer.size(), OTPointerFrozenSize(live_object));
  }

  /**
   * Takes an object that is laid out serially, and adjusts all its pointers
   * so that they are normalized to zero.
   */
  template<typename T>
  void PointerRefreeze(T* obj) {
    ot_private::OTPointerRefreezer fixer;
    fixer.InitBegin<T>(obj);
  }

  /**
   * Takes an object that is laid out serially with all its pointers
   * normalized to zero, and makes all the pointers live again.
   */
  template<typename T>
  T* PointerThaw(char *block) {
    ot_private::OTPointerThawer fixer;
    return fixer.InitBegin<T>(block);
  }
  
  /**
   * Takes an object that is laid out serially with all its pointers
   * normalized to zero, and makes all the pointers live again.
   */
  template<typename T>
  T* PointerRelocate(char *block, ptrdiff_t offset) {
    ot_private::OTPointerThawer fixer;
    return fixer.InitBegin<T>(block, offset);
  }
};

#endif
