// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * otrav_impl.h (this file not included by Doxygen)
 *
 * Default definitions for untraversed types, printer formats, and
 * traversal implementations.
 */

#ifndef NEED_OTRAV_IMPL
#error Include base/otrav.h instead of this file.
#endif

#include <typeinfo>



template<typename T>
inline void OT__BecomeNonAlias(T *ot__obj) {}

template<typename T>
inline void OT__BecomeAlias(T *ot__obj) {
  DEBUG_ONLY(NONFATAL(
      "Aliasing non-aliasable %s; may double-destruct.",
      typeid(T).name()));
}

template<typename T>
inline bool OT__IsAlias(T *ot__obj) { return false; }

template<typename T>
inline bool OT__Aliasable(const T *ot__obj) { return false; }



template<typename T>
inline void OT__DefaultConstruct(T *ot__obj) {
  if (OT__Aliasable(ot__obj)) {
    OT__BecomeAlias(ot__obj);
  }
}

template<typename T>
inline bool OT__DebugInitOK(T *ot__obj) {
  return !OT__Aliasable(ot__obj) || OT__IsAlias(ot__obj);
}

template<typename T>
inline bool OT__DebugModifyOK(T *ot__obj) {
  return !OT__Aliasable(ot__obj) || !OT__IsAlias(ot__obj);
}



template<typename T>
inline void OT__RefillTransients(T *ot__obj) {
  OT__BecomeNonAlias(ot__obj);
}

template<typename T, typename TVisitor>
inline void OT__TraverseTransients(T *ot__obj, TVisitor *ot__visitor) {}



template<typename T, typename TVisitor>
inline bool OT__PreTraverse(T *ot__dest, const T *ot__src,
			    index_t ot__len, TVisitor *ot__visitor) {
  return ot__visitor->PreUntraversed(ot__dest, ot__src, ot__len);
}

template<typename T, typename TVisitor>
inline void OT__TraverseObject(T *ot__obj, TVisitor *ot__visitor) {
  ot__visitor->Untraversed(ot__obj);
}



template<typename T>
inline bool OT__Shallow(const T *ot__obj) { return false; }

#define OT__MAKE_SHALLOW(T, TF) \
    template<> \
    inline bool OT__Shallow< T >(const T *ot__obj) { return true; }

FOR_ALL_PRIMITIVES_DO(OT__MAKE_SHALLOW)

template<typename T>
inline bool OT__Shallow(const T *const *ot__obj) { return true; }

#undef OT__MAKE_SHALLOW

template<typename T>
inline bool OT__ShallowOrPtr(T *const *ot__obj) { return true; }
template<typename T>
inline bool OT__ShallowOrPtr(const T *ot__obj) {
  return OT__Shallow(ot__obj);
}

template<typename T>
inline bool OT__NonConstPtr(const T *ot__obj) {
  return OT__ShallowOrPtr(ot__obj) && !OT__Shallow(ot__obj);
}



namespace ot {
  /**
   * Defines the print format exemplified by:
   * @code
   *   name : object_type =
   *     {
   *       member : primitive_type = value
   *       array : len n array_type =
   *         {
   *           [0] : elem_type = ...
   *           ...
   *         }
   *     }
   * @endcode
   */
  class StandardFormat {
    FORBID_ACCIDENTAL_COPIES(StandardFormat);

   private:
    FILE *stream_;
    int indent_;

   public:
    StandardFormat(FILE *stream) {
      stream_ = stream;
      indent_ = 0;
    }

   private:
    void PrintIndent_();

    void PrintHeader_(const char *name, index_t index,
		      const char *type, index_t len);

   public:
    void Untraversed(const unsigned char *obj, size_t bytes);

#define STANDARD_FORMAT__PRIMITIVE(T, TF) \
    void Primitive(const char *name, index_t index, \
                   const char *type, T val);

    FOR_ALL_PRIMITIVES_DO(STANDARD_FORMAT__PRIMITIVE)

#undef STANDARD_FORMAT__PRIMITIVE

    void Str(const char *name, index_t index,
	     const char *type, const char *str);

    void Ptr(const char *name, index_t index, 
	     const char *type, ptrdiff_t ptr);

    void Open(const char *name, index_t index, 
	      const char *type, index_t len = -1);

    void Close(const char *name, const char *type);
  };

  /**
   * Defines the print format exemplified by:
   * @code
   *   <object_type name="name">
   *     <primitive_type name="member">value</primitive_type>
   *     <array_type name="array" len="n">
   *       <elem_type index="0">...</elem_type>
   *       ...
   *     </array_type>
   *   </object_type>
   * @endcode
   */
  class XMLFormat {
    FORBID_ACCIDENTAL_COPIES(XMLFormat);

   private:
    FILE *stream_;
    int indent_;

   public:
    XMLFormat(FILE *stream) {
      stream_ = stream;
      indent_ = 0;
    }

   private:
    void PrintIndent_();

    void PrintHeader_(const char *name, index_t index,
		      const char *type, index_t len);

    void PrintFooter_(const char *type);

   public:
    void Untraversed(const unsigned char *obj, size_t bytes);

#define XML_FORMAT__PRIMITIVE(T, TF) \
    void Primitive(const char *name, index_t index, \
		   const char *type, T val);

    FOR_ALL_PRIMITIVES_DO(XML_FORMAT__PRIMITIVE)

#undef XML_FORMAT__PRIMITIVE

    void Str(const char *name, index_t index,
	     const char *type, const char *str);

    void Ptr(const char *name, index_t index,
	     const char *type, ptrdiff_t ptr);

    void Open(const char *name, index_t index, 
	      const char *type, index_t len = -1);

    void Close(const char *name, const char *type);
  };
};



namespace ot__private {
  /**
   * Traversal handler for pretty-printing objects.
   */
  template<typename TPrintFormat>
  class Printer {
    FORBID_ACCIDENTAL_COPIES(Printer)

   private:
    TPrintFormat format_;
    const char *name_;
    const char *type_;
    ptrdiff_t array_;
    index_t index_;

   public:
    enum { IS_PRINTER = 1 };

    template<typename T>
    Printer(const T &obj, const char *name, FILE *stream)
      : format_(stream) {
      name_ = name;
      type_ = typeid(obj).name();
      array_ = 0;
      index_ = -1;

      Obj(const_cast<T &>(obj));
    }

    template<typename T>
    void Elem(const T *array) {
      array_ = mem::PtrAbsAddr(array);
    }
    template<typename T>
    void Name(const char *name, const T &obj) {
      name_ = name;
      type_ = typeid(obj).name();
      if (array_) {
	index_ = &obj - reinterpret_cast<const T *>(array_);
	array_ = 0;
      } else {
	index_ = -1;
      }
    }

    template<typename T>
    bool PreUntraversed(T *dest, const T *src, index_t len) { return false; }
    template<typename T>
    bool PreTraverse(T *dest, const T *src, index_t len) { return false; }

    template<typename T>
    void Untraversed(T &obj) {
      format_.Untraversed(
          reinterpret_cast<const unsigned char *>(&obj), sizeof(T));
    }

#define PRINTER__PRIMITIVE_OBJ(T, TF) \
    void Obj(T val) { \
      format_.Primitive(name_, index_, type_, val); \
    }

    FOR_ALL_PRIMITIVES_DO(PRINTER__PRIMITIVE_OBJ)

#undef PRINTER__PRIMITIVE_OBJ

    void Obj(const char *str) {
      format_.Str(name_, index_, type_, str);
    }

    template<typename T>
    void Obj(T *ptr) {
      format_.Ptr(name_, index_, type_, mem::PtrAbsAddr(ptr));
    }

    template<typename T>
    void Obj(T &obj) {
      const char *name = name_;
      const char *type = type_;

      format_.Open(name, index_, type);
      OT__TraverseObject(&obj, this);
      format_.Close(name, type);
    }

    template<typename T>
    bool PreStaticArray(T *array, index_t len) {
      format_.Open(name_, index_, type_, len);
      return true;
    }
    template<typename T>
    void PostStaticArray(T *array, index_t len) {
      format_.Close(name_, type_);
    }

    template<typename T>
    bool PreArray(T *ptr, index_t len,
		  bool nullable, bool alloc, bool unitary) {
      if (nullable && ptr == NULL) {
	format_.Ptr(name_, index_, type_, 0);
	return false;
      } else {
	return PreStaticArray(ptr, len);
      }
    }
    template<typename T>
    void PostArray(T *ptr, index_t len,
		   bool nullable, bool alloc, bool unitary) {
      PostStaticArray(ptr, len);
    }

    void Str(const char *str, bool nullable, bool alloc) {
      format_.Str(name_, index_, type_, str);
    }
  };



  /**
   * Traversal handler for debug-poisoning objects' transients.
   */
  template<bool t_semi>
  class TransientUnstructor {
    FORBID_ACCIDENTAL_COPIES(TransientUnstructor)

   public:
    enum { IS_PRINTER = 0 };

    template<typename T>
    TransientUnstructor(T *obj) {
      OT__TraverseTransients(obj, this);
    }

    template<typename T>
    void Elem(const T *array) {}
    template<typename T>
    void Name(const char *name, const T &obj) {}

    template<typename T>
    bool PreUntraversed(T *dest, const T *src, index_t len) { return false; }
    template<typename T>
    bool PreTraverse(T *dest, const T *src, index_t len) { return false; }

    template<typename T>
    void Untraversed(T &obj) {}

    template<typename T>
    void Obj(T *&ptr) {
      DEBUG_POISON_PTR(ptr);
    }
    template<typename T>
    void Obj(T &obj) {
      if (OT__Shallow(&obj)) {
	mem::DebugPoison(&obj);
      } else if (t_semi) {
	mem::Construct(&obj);
      }
    }

    template<typename T>
    bool PreStaticArray(T *array, index_t len) {
      if (OT__Shallow(array)) {
	mem::DebugPoison(array, len);
      } else if (t_semi) {
	mem::Construct(array, len);
      }
      return false;
    }
    template<typename T>
    bool PreStaticArray(T **array, index_t len) {
      return true;
    }
    template<typename T>
    void PostStaticArray(T *array, index_t len) {}

    template<typename T>
    bool PreArray(T *&ptr, index_t len,
		  bool nullable, bool alloc, bool unitary) {
      DEBUG_ASSERT(!unitary || len == 1);
      if (nullable) {
        ptr = NULL;
      } else {
        DEBUG_POISON_PTR(ptr);
      }
      return false;
    }
    template<typename T>
    void PostArray(T *&ptr, index_t len,
                   bool nullable, bool alloc, bool unitary) {}

    void Str(char *&str, bool nullable, bool alloc) {
      if (nullable) {
        str = NULL;
      } else {
        DEBUG_POISON_PTR(str);
      }
    }
  };

  /**
   * Traversal handler for debug-poisoning objects and setting
   * nullable pointers to NULL.
   */
  class Unstructor {
    FORBID_ACCIDENTAL_COPIES(Unstructor)

   public:
    enum { IS_PRINTER = 0 };

    template<typename T>
    Unstructor(T *obj) {
      OT__TraverseObject(obj, this);
      TransientUnstructor<false> ot__unstructor(&obj);
    }

    template<typename T>
    void Elem(const T *array) {}
    template<typename T>
    void Name(const char *name, const T &obj) {}

    template<typename T>
    bool PreUntraversed(T *dest, const T *src, index_t len) { return false; }
    template<typename T>
    bool PreTraverse(T *dest, const T *src, index_t len) { return false; }

    template<typename T>
    void Untraversed(T &obj) {}

    template<typename T>
    void Obj(T *&ptr) {
      DEBUG_POISON_PTR(ptr);
    }
    template<typename T>
    void Obj(T &obj) {
      if (OT__Shallow(&obj)) {
	mem::DebugPoison(&obj);
      }
    }

    template<typename T>
    bool PreStaticArray(T *array, index_t len) {
      if (OT__Shallow(array)) {
	mem::DebugPoison(array, len);
      }
      return false;
    }
    template<typename T>
    bool PreStaticArray(T **array, index_t len) {
      return true;
    }
    template<typename T>
    void PostStaticArray(T *array, index_t len) {}

    template<typename T>
    bool PreArray(T *&ptr, index_t len,
		  bool nullable, bool alloc, bool unitary) {
      DEBUG_ASSERT(!unitary || len == 1);
      if (nullable) {
        ptr = NULL;
      } else {
        DEBUG_POISON_PTR(ptr);
      }
      return false;
    }
    template<typename T>
    void PostArray(T *&ptr, index_t len,
                   bool nullable, bool alloc, bool unitary) {}

    void Str(char *&str, bool nullable, bool alloc) {
      if (nullable) {
        str = NULL;
      } else {
        DEBUG_POISON_PTR(str);
      }
    }
  };

  /**
   * Traversal handler for freeing objects' transient memory.
   */
  template<bool t_semi>
  class TransientDestructor {
    FORBID_ACCIDENTAL_COPIES(TransientDestructor)

   public:
    enum { IS_PRINTER = 0 };

    template<typename T>
    TransientDestructor(T *obj) {
      OT__TraverseTransients(obj, this);
    }

    template<typename T>
    void Elem(const T *array) {}
    template<typename T>
    void Name(const char *name, const T &obj) {}

    template<typename T>
    bool PreUntraversed(T *dest, const T *src, index_t len) {
      if (OT__Shallow(dest)) {
	mem::DebugPoison(dest, len);
      }
      return OT__NonConstPtr(dest);
    }
    template<typename T>
    bool PreTraverse(T *dest, const T *src, index_t len) {
      return false;
    }

    template<typename T>
    void Untraversed(T &obj) {}

    template<typename T>
    void Obj(T &obj) {
      if (t_semi && !OT__ShallowOrPtr(&obj)) {
	mem::Destruct(&obj);
      }
    }

    template<typename T>
    bool PreStaticArray(T *array, index_t len) {
      return (t_semi && !OT__Shallow(array)) || OT__NonConstPtr(array);
    }
    template<typename T>
    void PostStaticArray(T *array, index_t len) {}

    template<typename T>
    bool PreArray(T *&ptr, index_t len,
		  bool nullable, bool alloc, bool unitary) {
      DEBUG_ASSERT(!unitary || len == 1);
      if (unlikely(!nullable || ptr != NULL)) {
        if (OT__PreTraverse(ptr, (T *)NULL, len, this)) {
	  return true;
	}
	PostArray(ptr, len, nullable, alloc, unitary);
      }
      return false;
    }
    template<typename T>
    void PostArray(T *&ptr, index_t len,
                   bool nullable, bool alloc, bool unitary) {
      if (alloc) {
	mem::FreeDestruct(ptr, len);
      } else if (unitary) {
	delete ptr;
      } else {
	delete[] ptr;
      }
      DEBUG_POISON_PTR(ptr);
    }

    void Str(char *&str, bool nullable, bool alloc) {
      if (unlikely(!nullable || str != NULL)) {
	DEBUG_ONLY(mem::DebugPoison(str, strlen(str) + 1));
	if (alloc) {
	  mem::Free(str);
	} else {
	  delete[] str;
	}
        DEBUG_POISON_PTR(str);
      }
    }
  };

  /**
   * Traversal handler for freeing objects' allocated memory.
   */
  template<bool t_semi>
  class Destructor {
    FORBID_ACCIDENTAL_COPIES(Destructor)

   public:
    enum { IS_PRINTER = 0 };

    template<typename T>
    Destructor(T *obj) {
      /* Optimized for faster case */
      if (unlikely(!OT__IsAlias(obj))) {
	TransientDestructor<t_semi> ot__destructor(obj);
	OT__TraverseObject(obj, this);
      }

      if (t_semi) {
	mem::DebugPoison(obj);
      } else {
#ifdef DEBUG
	Unstructor ot__unstructor(obj);
#endif
      }
    }

    template<typename T>
    void Elem(const T *array) {}
    template<typename T>
    void Name(const char *name, const T &obj) {}

    template<typename T>
    bool PreUntraversed(T *dest, const T *src, index_t len) {
      if (!t_semi && OT__Shallow(dest)) {
	mem::DebugPoison(dest, len);
      }
      return OT__NonConstPtr(dest);
    }
    template<typename T>
    bool PreTraverse(T *dest, const T *src, index_t len) {
      return t_semi && !OT__Shallow(dest);
    }

    template<typename T>
    void Untraversed(T &obj) {}

    template<typename T>
    void Obj(T &obj) {
      if (t_semi && !OT__ShallowOrPtr(&obj)) {
	TransientDestructor<t_semi> ot__destructor(&obj);
        OT__TraverseObject(&obj, this);
      }
    }

    template<typename T>
    bool PreStaticArray(T *array, index_t len) {
      return (t_semi && !OT__Shallow(array)) || OT__NonConstPtr(array);
    }
    template<typename T>
    void PostStaticArray(T *array, index_t len) {}

    template<typename T>
    bool PreArray(T *&ptr, index_t len,
		  bool nullable, bool alloc, bool unitary) {
      DEBUG_ASSERT(!unitary || len == 1);
      if (unlikely(!nullable || ptr != NULL)) {
        if (OT__PreTraverse(ptr, (T *)NULL, len, this)) {
	  return true;
	}
	PostArray(ptr, len, nullable, alloc, unitary);
      }
      return false;
    }
    template<typename T>
    void PostArray(T *&ptr, index_t len,
                   bool nullable, bool alloc, bool unitary) {
      if (t_semi) {
	mem::DebugPoison(ptr, len);
      } else {
	if (alloc) {
	  mem::FreeDestruct(ptr, len);
	} else if (unitary) {
	  delete ptr;
	} else {
	  delete[] ptr;
	}
      }
      DEBUG_POISON_PTR(ptr);
    }

    void Str(char *&str, bool nullable, bool alloc) {
      if (unlikely(!nullable || str != NULL)) {
	DEBUG_ONLY(mem::DebugPoison(str, strlen(str) + 1));
	if (alloc) {
	  mem::Free(str);
	} else {
	  delete[] str;
	}
        DEBUG_POISON_PTR(str);
      }
    }
  };

  /**
   * Traversal handler for recursively copying objects and their
   * contents, allocating memory as necessary.  Can copy from frozen
   * objects.
   */
  template<bool t_thawing>
  class Copier {
    FORBID_ACCIDENTAL_COPIES(Copier)

   private:
    ptrdiff_t offset_;

   public:
    enum { IS_PRINTER = 0 };

    template<typename T>
    Copier(T *dest, const T *src) {
      offset_ = mem::PtrAbsAddr(src);
      if (OT__PreTraverse(dest, src, 1, this)) {
	Obj(*dest);
      }
    }
    template<typename T>
    Copier(T *dest, const T *src, index_t len) {
      offset_ = mem::PtrAbsAddr(src);
      if (OT__PreTraverse(dest, src, len, this)) {
	for (index_t i = 0; i < len; ++i) {
	  Obj(dest[i]);
	}
      }
    }

    template<typename T>
    void Elem(const T *array) {}
    template<typename T>
    void Name(const char *name, const T &obj) {}

    template<typename T>
    bool PreUntraversed(T *dest, const T *src, index_t len) {
      mem::CopyConstruct(dest, src, len);
      return OT__NonConstPtr(dest);
    }
    template<typename T>
    bool PreTraverse(T *dest, const T *src, index_t len) {
      mem::Copy(dest, src, len);
      return !OT__Shallow(dest);
    }

    template<typename T>
    void Untraversed(T &obj) {
      char buf[sizeof(T)];
      mem::Copy<T, char, T>(buf, &obj);
      mem::CopyConstruct(&obj, reinterpret_cast<T *>(buf));
    }

    template<typename T>
    void Obj(T &obj) {
      if (!OT__ShallowOrPtr(&obj)) {
	OT__TraverseObject(&obj, this);
	TransientUnstructor<false> ot__unstructor(&obj);
	OT__RefillTransients(&obj);
      }
    }

    template<typename T>
    bool PreStaticArray(T *array, index_t len) {
      return !OT__Shallow(array);
    }
    template<typename T>
    void PostStaticArray(T *array, index_t len) {}

    template<typename T>
    bool PreArray(T *&ptr, index_t len,
		  bool nullable, bool alloc, bool unitary) {
      DEBUG_ASSERT(!unitary || len == 1);
      if (likely(nullable && ptr == NULL)) {
        return false;
      } else {
        const T *src = t_thawing ? mem::PtrAddBytes(ptr, offset_) : ptr;

	if (alloc) {
	  ptr = mem::Alloc<T>(len);
	} else if (unitary) {
	  ptr = new T;
	} else {
	  ptr = new T[len];
	}

	return OT__PreTraverse(ptr, src, len, this);
      }
    }
    template<typename T>
    void PostArray(T *&ptr, index_t len,
                   bool nullable, bool alloc, bool unitary) {}

    void Str(char *&str, bool nullable, bool alloc) {
      if (unlikely(!nullable || str != NULL)) {
        char *src = t_thawing ? mem::PtrAddBytes(str, offset_) : str;
	index_t len = strlen(src) + 1;

	if (alloc) {
	  str = mem::AllocCopy(src, len);
	} else {
	  str = mem::Copy(new char[len], src, len);
	}
      }
    }
  };

  /**
   * Traversal handler for recursively marking an object as an alias.
   */
  class Aliaser {
    FORBID_ACCIDENTAL_COPIES(Aliaser)

   public:
    enum { IS_PRINTER = 0 };

    template<typename T>
    Aliaser(T *obj) {
      Obj(*obj);
    }

    template<typename T>
    void Elem(const T *array) {}
    template<typename T>
    void Name(const char *name, const T &obj) {}

    template<typename T>
    bool PreUntraversed(T *dest, const T *src, index_t len) { return false; }
    template<typename T>
    bool PreTraverse(T *dest, const T *src, index_t len) { return false; }

    template<typename T>
    void Untraversed(T &obj) {
      DEBUG_ONLY(NONFATAL(
	  "Aliasing untraversed %s; may double-destruct.",
	  typeid(T).name()));
    }

    template<typename T>
    void Obj(T &obj) {
      if (!OT__ShallowOrPtr(&obj)) {
	OT__BecomeAlias(&obj);
	OT__TraverseObject(&obj, this);
	OT__TraverseTransients(&obj, this);
      }
    }

    template<typename T>
    bool PreStaticArray(T *array, index_t len) {
      return !OT__Shallow(array);
    }
    template<typename T>
    void PostStaticArray(T *array, index_t len) {}

    template<typename T>
    bool PreArray(T *&ptr, index_t len,
		  bool nullable, bool alloc, bool unitary) {
      return false;
    }
    template<typename T>
    void PostArray(T *&ptr, index_t len,
                   bool nullable, bool alloc, bool unitary) {}

    void Str(char *&str, bool nullable, bool alloc) {}
  };



  /**
   * Traversal handler for freezing objects and computing frozen size.
   */
  template<bool t_size_only>
  class Freezer {
    FORBID_ACCIDENTAL_COPIES(Freezer)

   private:
    char *block_;
    size_t pos_;

   public:
    enum { IS_PRINTER = 0 };

    template<typename T>
    Freezer(char *block, const T &src) {
      block_ = block;
      pos_ = sizeof(T);

      if (!t_size_only) {
        mem::Copy<T, char, T>(block, &src);
	Obj(*reinterpret_cast<T *>(block));
      } else {
	Obj(const_cast<T &>(src));
      }
    }

    size_t size() {
      return pos_;
    }

    template<typename T>
    void Elem(const T *array) {}
    template<typename T>
    void Name(const char *name, const T &obj) {}

    template<typename T>
    bool PreUntraversed(T *dest, const T *src, index_t len) { return false; }
    template<typename T>
    bool PreTraverse(T *dest, const T *src, index_t len) { return false; }

    template<typename T>
    void Untraversed(T &obj) {
      DEBUG_ONLY(NONFATAL(
	  "Freezing untraversed %s with bit-copy.",
          typeid(T).name()));
    }

    template<typename T>
    void Obj(T *&ptr) {
      DEBUG_ONLY(NONFATAL(
          "Freezing pointer %s as primitive.",
          typeid(T *).name()));
    }
    template<typename T>
    void Obj(T &obj) {
      if (!OT__Shallow(&obj)) {
	OT__TraverseObject(&obj, this);
      }
    }

    template<typename T>
    bool PreStaticArray(T *array, index_t len) {
      return !OT__Shallow(array);
    }
    template<typename T>
    void PostStaticArray(T *array, index_t len) {}

    template<typename T>
    bool PreArray(T *&ptr, index_t len,
		  bool nullable, bool alloc, bool unitary) {
      DEBUG_ASSERT(!unitary || len == 1);
      if (likely(nullable && ptr == NULL)) {
        return false;
      } else {
        pos_ = stride_align(pos_, T);

        if (!t_size_only) {
          T *dest = reinterpret_cast<T *>(mem::PtrAddBytes(block_, pos_));
          mem::Copy(dest, ptr, len);
          ptr = OT__Shallow(ptr) ? reinterpret_cast<T *>(pos_) : dest;
        }

        pos_ += sizeof(T) * len;
        return !OT__Shallow(ptr);
      }
    }
    template<typename T>
    void PostArray(T *&ptr, index_t len,
                   bool nullable, bool alloc, bool unitary) {
      if (!t_size_only) {
        ptr = reinterpret_cast<T *>(mem::PtrDiffBytes(ptr, block_));
      }
    }

    void Str(char *&str, bool nullable, bool alloc) {
      if (unlikely(!nullable || str != NULL)) {
	index_t len = strlen(str) + 1;
        pos_ = stride_align(pos_, char);

        if (!t_size_only) {
          char *dest = mem::PtrAddBytes(block_, pos_);
          mem::Copy(dest, str, len);
          str = reinterpret_cast<char *>(pos_);
        }

        pos_ += sizeof(char) * len;
      }
    }
  };

  /**
   * Traversal handler for moving frozen objects.  Also can re-freeze,
   * semi-thaw, and semi-copy.
   */
  template<bool t_freezing, bool t_transients>
  class Relocator {
    FORBID_ACCIDENTAL_COPIES(Relocator)

   private:
    const char *block_;
    const char *orig_;
    ptrdiff_t offset_;

   public:
    enum { IS_PRINTER = 0 };

    template<typename T>
    Relocator(char *block, const T *orig) {
      block_ = block;
      orig_ = reinterpret_cast<const char *>(orig);
      offset_ = mem::PtrDiffBytes(block, orig);

      Obj(*reinterpret_cast<T *>(block));
    }

    template<typename T>
    void Elem(const T *array) {}
    template<typename T>
    void Name(const char *name, const T &obj) {}

    template<typename T>
    bool PreUntraversed(T *dest, const T *src, index_t len) { return false; }
    template<typename T>
    bool PreTraverse(T *dest, const T *src, index_t len) { return false; }

    template<typename T>
    void Untraversed(T &obj) {}

    template<typename T>
    void Obj(T &obj) {
      if (!OT__ShallowOrPtr(&obj)) {
	if (t_freezing && t_transients) {
	  TransientDestructor<true> ot__destructor(&obj);
	}
	OT__TraverseObject(&obj, this);
	if (!t_freezing && t_transients) {
	  TransientUnstructor<true> ot__unstructor(&obj);
	  OT__RefillTransients(&obj);
	}
      }
    }

    template<typename T>
    bool PreStaticArray(T *array, index_t len) {
      return !OT__Shallow(array);
    }
    template<typename T>
    void PostStaticArray(T *array, index_t len) {}

    template<typename T>
    bool PreArray(T *&ptr, index_t len,
		  bool nullable, bool alloc, bool unitary) {
      DEBUG_ASSERT(!unitary || len == 1);
      if (likely(nullable && ptr == NULL)) {
        return false;
      } else if (t_freezing && OT__Shallow(ptr)) {
        ptr = reinterpret_cast<T *>(mem::PtrDiffBytes(ptr, orig_));
        return false;
      } else {
        ptr = mem::PtrAddBytes(ptr, offset_);
        return !OT__Shallow(ptr);
      }
    }
    template<typename T>
    void PostArray(T *&ptr, index_t len,
                   bool nullable, bool alloc, bool unitary) {
      if (t_freezing) {
        ptr = reinterpret_cast<T *>(mem::PtrDiffBytes(ptr, block_));
      }
    }

    void Str(char *&str, bool nullable, bool alloc) {
      if (unlikely(!nullable || str != NULL)) {
	if (t_freezing) {
	  str = reinterpret_cast<char *>(mem::PtrDiffBytes(str, orig_));
	} else {
	  str = mem::PtrAddBytes(str, offset_);
	}
      }
    }
  };



  /**
   * Traversal handler for serializing objects or computing serial
   * size.
   */
  template<bool t_size_only>
  class Serializer {
    FORBID_ACCIDENTAL_COPIES(Serializer)

   private:
    FILE *stream_;
    size_t size_;

    template<typename T>
    void Write_(T *val, index_t elems = 1) {
      if (!t_size_only) {
	fwrite(val, sizeof(T), elems, stream_);
      }
      size += sizeof(T) * elems;
    }

   public:
    enum { IS_PRINTER = 0 };

    template<typename T>
    Serializer(const T &obj, FILE *stream) {
      stream_ = stream;
      size_ = 0;

      Obj(const_cast<T *>(&obj));
    }

    size_t size() {
      return size_;
    }

    template<typename T>
    void Elem(const T *array) {}
    template<typename T>
    void Name(const char *name, const T &obj) {}

    template<typename T>
    bool PreUntraversed(T **dest, const T **src, index_t len) {
      if (OT__NonConstPtr(src)) {
	return true;
      } else {
	DEBUG_WARN_MSG_IF(!t_size_only,
            "Serializing pointer %s as primitive.",
            typeid(T *).name());
	Write_(src, len);
	return false;
      }
    }
    template<typename T>
    bool PreUntraversed(T *dest, const T *src, index_t len) {
      DEBUG_WARN_MSG_IF(!t_size_only && !OT__Shallow(src),
          "Serializing untraversed %s with bit-copy.",
          typeid(T).name());
      Write_(src, len);
      return false;
    }
    template<typename T>
    bool PreTraverse(T *dest, const T *src, index_t len) {
      if (OT__IsShallow(src)) {
	Write_(src, len);
	return false;
      } else {
	return true;
      }
    }

    template<typename T>
    void Untraversed(T &obj) {
      DEBUG_WARN_MSG_IF(!t_size_only && !OT__Shallow(&obj),
          "Serializing untraversed %s with bit-copy.",
          typeid(T).name());
      Write_(&obj);
    }

    template<typename T>
    void Obj(T *&ptr) {
      DEBUG_WARN_MSG_IF(!t_size_only,
          "Serializing pointer %s as primitive.",
          typeid(T *).name());
      Write_(&ptr);
    }
    template<typename T>
    void Obj(T &obj) {
      if (OT__Shallow(&obj)) {
	Write_(&obj);
      } else {
	OT__TraverseObject(&obj, this);
      }
    }

    template<typename T>
    bool PreStaticArray(T *array, index_t len) {
      return OT__PreTraverse((T *)NULL, array, len, this);
    }
    template<typename T>
    void PostStaticArray(T *array, index_t len) {}

    template<typename T>
    bool PreArray(T *&ptr, index_t len,
		  bool nullable, bool alloc, bool unitary) {
      DEBUG_ASSERT(!unitary || len == 1);
      if (nullable) {
	bool temp = ptr != NULL;
	Write_(&temp);
	if (likely(!temp)) {
	  return false;
	}
      }
      return OT__PreTraverse((T *)NULL, ptr, len, this);
    }
    template<typename T>
    void PostArray(T *&ptr, index_t len,
                   bool nullable, bool alloc, bool unitary) {}

    void Str(char *&str, bool nullable, bool alloc) {
      if (nullable) {
	bool temp = str != NULL;
	Write_(&temp);
	if (likely(!temp)) {
	  return;
	}
      }
      Write_(str, strlen(str) + 1);
    }
  };

  /**
   * Traversal handler for deserializing objects.
   */
  class Deserializer {
    FORBID_ACCIDENTAL_COPIES(Deserializer)

   private:
    FILE *stream_;
    size_t size_;

    template<typename T>
    void Read_(T *ptr, index_t elems = 1) {
      fread(ptr, sizeof(T), elems, stream_);
      size_ += sizeof(T) * elems;
    }

    index_t ReadStrLen_() {
      index_t len = 0;
      fpos_t pos;

      fgetpos(stream_, &pos);
      while (getc(stream_)) {
	++len;
      }
      fsetpos(stream_, &pos);

      return len;
    }

   public:
    enum { IS_PRINTER = 0 };

    template<typename T>
    Deserializer(T &obj, FILE *stream) {
      stream_ = stream;
      size_ = 0;

      Obj(&obj);
    }

    size_t size() {
      return size_;
    }

    template<typename T>
    void Elem(const T *array) {}
    template<typename T>
    void Name(const char *name, const T &obj) {}

    template<typename T>
    bool PreUntraversed(T **dest, const T **src, index_t len) {
      if (OT__NonConstPtr(dest)) {
	return true;
      } else {
	Read_(dest, len);
	return false;
      }
    }
    template<typename T>
    bool PreUntraversed(T *dest, const T *src, index_t len) {
      Read_(dest, len);
      return false;
    }
    template<typename T>
    bool PreTraverse(T *dest, const T *src, index_t len) {
      if (OT__IsShallow(src)) {
	Read_(dest, len);
	return false;
      } else {
	return true;
      }
    }

    template<typename T>
    void Untraversed(T &obj) {
      Read_(&obj);
    }

    template<typename T>
    void Obj(T &obj) {
      if (OT__ShallowOrPtr(&obj)) {
	Read_(&obj);
      } else {
	OT__TraverseObject(&obj, this);
	TransientUnstructor<false> ot__unstructor(&obj);
	OT__RefillTransients(&obj);
      }
    }

    template<typename T>
    bool PreStaticArray(T *array, index_t len) {
      return OT__PreTraverse(array, (T *)NULL, len, this);
    }
    template<typename T>
    void PostStaticArray(T *array, index_t len) {}

    template<typename T>
    bool PreArray(T *&ptr, index_t len,
		  bool nullable, bool alloc, bool unitary) {
      DEBUG_ASSERT(!unitary || len == 1);
      if (nullable) {
	bool temp;
	Read_(&temp);
	if (likely(!temp)) {
	  ptr = NULL;
	  return false;
	}
      }
      if (alloc) {
	ptr = mem::Alloc<T>(len);
      } else if (unitary) {
	ptr = new T;
      } else {
	ptr = new T[len];
      }
      return OT__PreTraverse(ptr, (T *)NULL, len, this);
    }
    template<typename T>
    void PostArray(T *&ptr, index_t len) {}

    void Str(char *&str, bool nullable, bool alloc) {
      if (nullable) {
	bool temp;
	Read_(&temp);
	if (likely(!temp)) {
	  str = NULL;
	  return;
	}
      }
      index_t len = ReadStrLen_() + 1;
      if (alloc) {
	str = mem::Alloc<char>(len);
      } else {
	str = new char[len];
      }
      Read_(str, len);
    }
  };
};
