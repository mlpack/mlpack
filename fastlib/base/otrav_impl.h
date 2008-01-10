/* Template implementations for object-traversal. */

// Note since this has no "file" tag it won't get generated as documentation.

// Private namespace for object-traversal utilities.
namespace ot__private {
  // TODO: Space-conservatory serialization and deserialization
  // (Currently only freezing/thawing is supported)

  template<typename DefaultPrinter, typename Printer, typename T>
  struct ZOTPrinter_Dispatcher {
    static void Print(const char *name, T& x, Printer* printer) {
      DefaultPrinter::Print(name, x, printer);
    }
  };

  /* macro for use within this file */
  #define OTPRINTER__SPECIAL(T, format_str) \
    template<typename DefaultPrinter, typename Printer> \
    struct ZOTPrinter_Dispatcher<DefaultPrinter, Printer, T> { \
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
  OTPRINTER__SPECIAL(float, "float = %g");
  OTPRINTER__SPECIAL(double, "double = %g");

  /* Utility class to take an OT-compatible object and prints it to screen. */
  class ZOTPrinter {
   private:
    FILE *stream_;
    int indent_amount_;
    const char *name_;

   private:
    template<typename T>
    struct DefaultPrimitivePrinter {
      static void Print(const char *name, const T& x, ZOTPrinter *printer) {
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
      static void Print(const char *name, T& x, ZOTPrinter *printer) {
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

    /* Stores the name of the object going to come in. */
    void Name(const char *s) {
      name_ = s;
    }

    template<typename T> void Primitive(T& x) {
      ZOTPrinter_Dispatcher< DefaultPrimitivePrinter<T>, ZOTPrinter, T >
          ::Print(name_, x, this);
    }

    template<typename T> void Object(T* obj, bool nullable,
        const char *label) {
      if (nullable && !obj) {
        Write("%s : %s %s = NULL", name_, label, typeid(T).name());
      } else {
        ZOTPrinter_Dispatcher< DefaultObjectPrinter<T>, ZOTPrinter, T >
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

    /* Visits an internal object. */
    template<typename T> void MyObject(T& x) {
      // Recurse in case this sub-object has pointers
      Object(&x, false, "embedded");
    }
    /* Visits an array. */
    template<typename T> void MyArray(T* x, index_t len) {
      // Recurse in case any of these objects have pointers
      Array(x, len);
    }

    /*
     * Visits an object pointed to, allocated with new.
     *
     * This allocates space within the block for the pointer, copies the
     * data pointed to, and recurses on the data pointed to.
     */
    template<typename T> void Ptr(T*& source_region, bool nullable) {
      Object(source_region, nullable, "pointer-to");
    }

    /* Visits an array pointed to, allocated with malloc */
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

  /*
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
   * TODO: Consider making frozen points relative to the pointer's address
   * rather than relative to the base address.
   *
   * ANY MODIFICATIONS TO THIS MUST ALSO BE MADE TO THE SIZE CALCULATOR!
   */
  class ZOTPointerFreezer {
   private:
    /* The block of memory to freeze into. */
    char *block_;
    /* The current position within the block. */
    ptrdiff_t pos_;
    /*
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
      freeze_offset_ = mem::PtrDiffBytes(block_, &x);

      mem::BitCopy(reinterpret_cast<T*>(block_), &x);
      // we must cast away const due to TraverseObject's limitations
      TraverseObject(const_cast<T*>(&x), this);
    }

    size_t size() const {
      return stride_align_max(pos_);
    }

    /* Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /* Visits an object with no OT implementation. */
    template<typename T> void Primitive(T& x) {
      // Primitives can be bit-copied
    }

    /* Visits an internal object. */
    template<typename T> void MyObject(T& x) {
      // Recurse in case this sub-object has pointers
      TraverseObject(&x, this);
    }
    /* Visits an array. */
    template<typename T> void MyArray(T* x, index_t len) {
      // Recurse in case any of these objects have pointers
      TraverseArray(x, len, this);
    }

    /*
     * Visits an object pointed to, allocated with new.
     *
     * This allocates space within the block for the pointer, copies the
     * data pointed to, and recurses on the data pointed to.
     */
    template<typename T> void Ptr(T*& source_region, bool nullable);

    /*
     * Visits an array pointed to, allocated with new[].
     *
     * This allocates space within the block for the array, copies the
     * data pointed to, and recurses on the array's elements.
     */
    template<typename T> void Array(T*& source_region, index_t len);

    /* Visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& source_region, index_t len) {
      Array(source_region, len);
    }

   private:
    template <typename T>
    /*
     * Gets a pointer to the pointer in the destination region that needs
     * to be updated.  A picture might help.
     *
     * @param source_region_ptr the pointer to the original pointer, in
     *        its original location within the larger structure, used with
     *        pointer arithmetic for updating the resulting pointers
     */
    T** DestinationEquivalentPointer_(T** source_region_ptr) {
      return mem::PtrAddBytes(source_region_ptr, freeze_offset_);
    }
    /*
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

  template<typename T> void ZOTPointerFreezer::Ptr(
      T*& source_region, bool nullable) {
    if (nullable && unlikely(source_region == NULL)) {
      *DestinationEquivalentPointer_(&source_region) = NULL;
    } else {
      // Get the pointer we will write into, and fix our internal pointer
      T* dest = TranslateAndFixPointer_(&source_region);
      // Copy the object and progress
      pos_ += sizeof(T);
      mem::BitCopy(dest, source_region);
      // Save our old freeze offset
      size_t freeze_offset_tmp = freeze_offset_;
      // Calculate new freeze offset as the distance between the source and
      // destination memory regions.
      freeze_offset_ = mem::PtrDiffBytes(dest, source_region);
      // Recurse on the object.
      TraverseObject(source_region, this);
      // Revert to the old freeze offset.
      freeze_offset_ = freeze_offset_tmp;
    }
  }

  template<typename T> void ZOTPointerFreezer::Array(
      T*& source_region, index_t len) {
    if (len == 0) {
      *DestinationEquivalentPointer_(&source_region) = NULL;
    } else {
      // Get the pointer we will write into, and fix our internal pointer
      T* dest = TranslateAndFixPointer_(&source_region);
      // Calculate the total size allocated, copy, and progress
      size_t size = len * sizeof(T);
      pos_ += size;
      mem::BitCopyBytes(dest, source_region, size);
      // Save old freeze offset
      size_t freeze_offset_tmp = freeze_offset_;
      // Calculate new freeze offset
      freeze_offset_ = mem::PtrDiffBytes(dest, source_region);
      // Recurse over each object
      for (index_t i = 0; i < len; i++) {
        TraverseObject(&source_region[i], this);
      }
      // Restore old freeze offset because we have returned to the old object
      freeze_offset_ = freeze_offset_tmp;
    }
  }

  class ZOTFrozenSizeCalculator {
   private:
    size_t pos_;

   public:
    template<typename T>
    void Doit(const T& obj) {
      pos_ = sizeof(T);
      TraverseObject(const_cast<T*>(&obj), this);
    }

    /*
     * Returns the calculated size.
     */
    size_t size() const {
      return stride_align_max(pos_);
    }

    /* Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /* visits an object with no OT implementation */
    template<typename T> void Primitive(T& x) {}
    /* visits an internal object */
    template<typename T> void MyObject(T& x) {
      TraverseObject(&x, this);
    }
    /* visits an array */
    template<typename T> void MyArray(T* x, index_t len) {
      TraverseArray(x, len, this);
    }

    /* visits an object pointed to, allocated with new */
    template<typename T> void Ptr(T*& x, bool nullable) {
      if (!nullable || x != NULL) {
        PretendLayout_<T>(1);
        TraverseObject(x, this);
      }
    }
    /* visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len) {
      if (len != 0) {
        PretendLayout_<T>(len);
        TraverseArray(x, len, this);
      }
    }
    /* visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len) {
      Array(x, len);
    }

   private:
    template<typename T>
    void PretendLayout_(index_t count) {
      pos_ = (stride_align(pos_, T)) + (sizeof(T) * count);
    }
  };

  class ZOTPointerThawer {
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

    /* Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /* visits an object with no OT implementation */
    template<typename T> void Primitive(T& x) {}
    /* visits an internal object */
    template<typename T> void MyObject(T& x) {
      TraverseObject(&x, this);
      TraverseObjectPostprocess(&x);
    }
    /* visits an array */
    template<typename T> void MyArray(T* x, index_t len) {
      for (index_t i = 0; i < len; i++) {
        MyObject(x[i]);
      }
    }
    /* visits an object pointed to, allocated with new */
    template<typename T> void Ptr(T*& x, bool nullable) {
      if (!nullable || x != NULL) {
        x = mem::PtrAddBytes(x, offset_);
        MyObject(*x);
      }
    }
    /* visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len) {
      if (len != 0) {
        x = mem::PtrAddBytes(x, offset_);
        MyArray(x, len);
      }
    }
    /* visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len) {
      Array(x, len);
    }
  };

  class ZOTPointerRelocator {
   private:
    ptrdiff_t pre_offset_;
    ptrdiff_t post_offset_;

   public:
    /*
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

    /* Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /* visits an object with no OT implementation */
    template<typename T> void Primitive(T& x) {}
    /* visits an internal object */
    template<typename T> void MyObject(T& x) {
      TraverseObject(&x, this);
    }
    /* visits an array */
    template<typename T> void MyArray(T* x, index_t len) {
      TraverseArray(x, len, this);
    }
    /* visits an object pointed to, allocated with new */
    template<typename T> void Ptr(T*& x, bool nullable) {
      if (!nullable || x != NULL) {
        TraverseObject(mem::PtrAddBytes(x, pre_offset_), this);
        x = mem::PtrAddBytes(x, post_offset_);
      }
    }
    /* visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len) {
      if (len != 0) {
        TraverseArray(mem::PtrAddBytes(x, pre_offset_), len, this);
        x = mem::PtrAddBytes(x, post_offset_);
      }
    }
    /* visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len) {
      Array(x, len);
    }
  };

  struct ZOTDeepCopier {
   public:
    template<typename T>
    static void Doit(const T& src, T *dest) {
      ot__private::ZOTDeepCopier d;
      mem::BitCopy(dest, &src, 1);
      TraverseObjectPostprocess(dest);
      d.MyObject(*dest);
    }

    /* Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /* visits an object with no OT implementation */
    template<typename T> void Primitive(T& x) {}
    /* visits an internal object */
    template<typename T> void MyObject(T& x) {
      TraverseObject(&x, this);
      TraverseObjectPostprocess(&x);
    }
    /* visits an array */
    template<typename T> void MyArray(T* x, index_t len) {
      TraverseArray(x, len, this);
    }
    /* visits an object pointed to, allocated with new */
    template<typename T> void Ptr(T*& x, bool nullable) {
      if (!nullable || x != NULL) {
        x = new T(*x);
      }
    }
    /* visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len) {
      x = mem::CopyConstruct(new T[len], x, len);
    }
    /* visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len) {
      x = mem::CopyConstruct(mem::Alloc<T>(len), x, len);
    }
  };

  struct ZOTDestructor {
   public:
    /* Receives the nanme of the upcoming object -- we ignore this. */
    void Name(const char *s) {}

    /* visits an object with no OT implementation */
    template<typename T> void Primitive(T& x) {}
    /* visits an internal object */
    template<typename T> void MyObject(T& x) {
      // C++ will automatically chain this
    }
    /* visits an array */
    template<typename T> void MyArray(T* x, index_t len) {
      // C++ will automatically chain this
    }
    /* visits an object pointed to, allocated with new */
    template<typename T> void Ptr(T*& x, bool nullable) {
      if (!nullable || x != NULL) {
        delete x;
      }
      DEBUG_POISON_PTR(x);
    }
    /* visits an array pointed to, allocated with new[] */
    template<typename T> void Array(T*& x, index_t len, bool nullable) {
      delete[] x;
      DEBUG_POISON_PTR(x);
    }
    /* visits an array pointed to, allocated with malloc */
    template<typename T> void MallocArray(T*& x, index_t len) {
      T *tmpx = x;
      mem::Destruct(tmpx, len);
      mem::Free(tmpx);
      DEBUG_POISON_PTR(x);
    }
  };

  template<typename T>
  void DestructorImplementation(T *dest) {
    ZOTDestructor d;
    TraverseObject(dest, &d);
    // can't poison this because of destructor chanining
  }
}; // namespace ot__private
