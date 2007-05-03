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

#define PASADENA(x) 3
// TODO: Remove nullability from arrays

/**
 * Within OT_DEF, declare a sub-object (or primitive) that is directly
 * contained, NOT pointed to.
 */
#define OT_MY_OBJECT(x) v_OT->MyObject(this->x)
/**
 * Within OT_DEF, declare a static-sized array embedded within your object.
 *
 * The length of the array is determined automatically via sizeof.
 */
#define OT_MY_ARRAY(x) v_OT->MyArray(this->x, sizeof(this->x) / sizeof(this->x[0]))
/**
 * Within OT_DEF, declare an object being pointed to, managed by
 * new and delete.
 */
#define OT_PTR(x) v_OT->Ptr(this->x, false)
/**
 * Within OT_DEF, declare an array being pointed to, managed by
 * new[] and delete[].
 */
#define OT_ARRAY(x, i) v_OT->Array(this->x, i, false)
/**
 * Within OT_DEF, declare an array or object being pointed to managed by
 * malloc and free.
 */
#define OT_MALLOC_ARRAY(x, i) v_OT->MallocArray(this->x, i, false)
/**
 * Within OT_DEF, declare a pointer to an object that might be NULL.
 */
#define OT_PTR_NULLABLE(x) v_OT->Ptr(this->x, true)
/**
 * Within OT_DEF, declare a pointer to an array that might be NULL.
 */
#define OT_ARRAY_NULLABLE(x, i) v_OT->Array(this->x, i, true)
/**
 * Within OT_DEF, declare a pointer to a malloced array that might be NULL.
 */
#define OT_MALLOC_ARRAY_NULLABLE(x, i) v_OT->MallocArray(this->x, i, true)

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
  // TODO: Conservatory serialization and deserialization
  
  /** Visits an object with no OT implementation. */
  template<typename Printer, typename T> inline void OTPrinter_Primitive(const T& x, Printer* printer) {
    printer->Write("%s (don't know how to print)", typeinfo(x).name());
  }
  template<typename Printer> inline void OTPrinter_Primitive(const char* x, Printer* printer) {
    printer->Write("string %s", x);
  }
  template<typename Printer> inline void OTPrinter_Primitive(char x, Printer* printer) {
    printer->Write("char %d", x);
  }
  template<typename Printer> inline void OTPrinter_Primitive(short x, Printer* printer) {
    printer->Write("short %d", x);
  }
  template<typename Printer> inline void OTPrinter_Primitive(int x, Printer* printer) {
    printer->Write("int %d", x);
  }
  template<typename Printer> inline void OTPrinter_Primitive(long x, Printer* printer) {
    printer->Write("long %ld", x);
  }
  template<typename Printer> inline void OTPrinter_Primitive(unsigned char x, Printer* printer) {
    printer->Write("uchar %u", x);
  }
  template<typename Printer> inline void OTPrinter_Primitive(unsigned short x, Printer* printer) {
    printer->Write("ushort %u", x);
  }
  template<typename Printer> inline void OTPrinter_Primitive(unsigned int x, Printer* printer) {
    printer->Write("uint %u", x);
  }
  template<typename Printer> inline void OTPrinter_Primitive(unsigned long x, Printer* printer) {
    printer->Write("ulong %lu", x);
  }
  template<typename Printer> inline void OTPrinter_Primitive(float x, Printer* printer) {
    printer->Write("float %f", x);
  }
  template<typename Printer> inline void OTPrinter_Primitive(double x, Printer* printer) {
    printer->Write("double %f", x);
  }

  /**
   * Takes an OT-compatible object and prints it to screen.
   */
  class OTPrinter {
   private:
    FILE *stream_;
    int indent_amount_;
    
   public:
    template<typename T>
    void InitBegin(const T& x, FILE *stream_in) const {
      stream_ = stream_in;
      indent_amount_ = 0;
      TraverseObject(const_cast<T*>(&x), this);
    }
    
    template<typename T> inline void Primitive(const T& x) {
      OTPrinter_Primitive(x, this);
    }

    template<typename T> void Object(T* obj, bool nullable) {
      if (nullable && !obj) {
        Write("object %s NULL {}", typeid(T).name());
      } else {
        Indent(2);
        Write("object %s {", typeid(T).name());
        TraverseObject(obj, this);
        Write("} end object %s", typeid(T).name());
        Indent(-2);
      }
    }

    template<typename T> void Array(T* array, index_t len,
        bool nullable) {
      if (nullable && !array) {
        Write("array %s NULL {}", typeid(T).name());
      } else {
        Indent(2);
        Write("array %s len %"LI"d {", typeid(T).name(), len);
        TraverseObject(array, this);
        Write("} end array %s", typeid(T).name());
        Indent(-2);
      }
    }

    /** Visits an internal object. */
    template<typename T> void MyObject(T& x) {
      // Recurse in case this sub-object has pointers
      Object(&x);
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
      Object(source_region, nullable);
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
    void InitBegin(char *block_in, const T& x) {
      block_ = block_in;
      pos_ = sizeof(T);
      freeze_offset_ = PointerDiff(block_, &x);
      
      mem::Copy(mem::PointerAbsoluteAddress(block_), &x);
      // we must cast away const due to TraverseObject's limitations
      TraverseObject(const_cast<T*>(&x), this);
    }
    
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
    T* DestinationEquivalentPointer_(T** source_region_ptr) {
      return PointerAdd(source_region_ptr, freeze_offset_);
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
      DEBUG_ASSERT(*pointer_to_fix == *source_region_ptr);
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
      freeze_offset_ = PointerDiff(dest, source_region);
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
      mem::CopyBytes(reinterpret_cast<T*>(block_ + pos_), source_region, size);
      // Save old freeze offset
      size_t freeze_offset_tmp = freeze_offset_;
      // Calculate new freeze offset
      freeze_offset_ = PointerDiff(dest, source_region);
      // Recurse over each object
      for (index_t i = 0; i < len; i++) {
        T* dest_array_element = &dest[i];
        TraverseObject(dest_array_element);
        TraverseObjectPostprocess(dest_array_element);
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
      TraverseObject(const_cast<T*>(&obj), this);
    }

    /**
     * Returns the calculated size.
     */
    size_t size() const {
      return pos_;
    }
    
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
      pos_ = stride_align(pos_, T) + sizeof(T) * count;
    }
  };

  class OTPointerThawer {
   private:
    ptrdiff_t offset_;
    
   public:
    template<typename T>
    void InitBegin(char *data) {
      offset_ = reinterpret_cast<size_t>(data);
      TraverseObject(reinterpret_cast<T*>(data), this);
    }
    
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
    void InitBegin(char *data) {
      offset_ = -reinterpret_cast<size_t>(data);
      TraverseObject(reinterpret_cast<T*>(data), this);
    }
    
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

template<typename T>
void OTPrint(const T& object, FILE *stream) {
  ot_private::OTPrinter printer;
  printer.InitBegin(object, stream);
}

/**
 * Makes a copy of an object, freezing it for the first time.
 */
template<typename T>
T* OTPointerFreeze(const T& live_object, char *block) {
  ot_private::OTPointerFreezer freezer;
  freezer.InitBegin(live_object, block);
}

/**
 * Takes an object that is laid out serially, and adjusts all its pointers
 * so that they are normalized to zero.
 */
template<typename T>
T* OTPointerRefreeze(char *block) {
  ot_private::OTPointerRefreezer fixer;
  fixer.InitBegin<T>(block);
}

/**
 * Takes an object that is laid out serially with all its pointers
 * normalized to zero, and makes all the pointers live again.
 */
template<typename T>
T* OTPointerThaw(char *block) {
  ot_private::OTPointerThawer fixer;
  fixer.InitBegin<T>(block);
}

#endif
