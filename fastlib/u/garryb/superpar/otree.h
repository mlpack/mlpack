
#define USE_OT(AClass, visitor) \
 public: \
  template<typename Visitor> \
  friend void TraverseObject(AClass *x, Visitor *v) { \
    x->TraverseObject(v); \
  } \
  \
 private: \
  template<typename Visitor> \
  void TraverseObject_(Visitor *visitor_variable_name) \

#define USE_OT_FULL(AClass, visitor_variable_name) \
 public: \
  ~AClass() { \
    OTDestructorVisitor v; \
    TraverseObject_(&v); \
  } \
  \
  AClass(const AClass& other) { \
    OTCopyConstructorVisitor v(other); \
    TraverseObject_(&v); \
  } \
  \
  const AClass& operator = (const AClass& other) { \
    OTAssignmentVisitor v(other); \
    TraverseObject_(&v); \
  } \
  \
  USE_OT(AClass, visitor)


class OTVisitor {
 public:
  template<typename T> void Mine(T& x);
  template<typename T> void Ptr(T*& x);
  template<typename T> void Array(T*& x, int i);
  template<typename T> void MallocPtr(T*& x);
  template<typename T> void MallocArray(T*& x, int i);
};

template<typename X, typename Visitor>
void TraverseObject(X* x, Visitor* v) {
  v->Primitive(*x);
}

class OTDestructorVisitor {
 public:
  template<typename T> void Primitive(T& x) {}
  template<typename T> void Mine(T& x) {
    // not necessary for destructors - destructors chain automatically
  }
  template<typename T> void Ptr(T*& x) {
    delete x;
  }
  template<typename T> void Array(T*& x, int i) {
    delete[] x;
  }
  template<typename T> void MallocPtr(T*& x) {
    x->~T(); free(x); DEBUG_POISON_PTR(x);
  }
  template<typename T> void MallocArray(T*& x, int i) {
    mem::DestructAll(x, t); free(x); DEBUG_POISON_PTR(x);
  }
};

class OTCopyConstructorVisitor {
 public:
  template<typename T> void Primitive(T& x) {
    *x = *GetPointer(x);
  }
  template<typename T> void Mine(T& x) {
    TraverseObject(x, this);
  }
  template<typename T> void Ptr(T*& x) {
    recursively copy x
  }
  template<typename T> void Array(T*& x, int i) {
    recursively copy x
  }
  template<typename T> void MallocPtr(T*& x) {
    recursively copy x
  }
  template<typename T> void MallocArray(T*& x, int i) {
    recursively copy x
  }
};

class OTDestructorVisitor {
  void 
};


class AClass {
  Foo a;
  Bar b;
  
  USE_OT(AClass, v) {
    v->Mine(a);
    v->Mine(b);
  }
};

class Vector {
  double *d;
  index_t len;
  
  USE_OT(Vector, v) {
    v->Mine(len);
    v->MallocPtr(d, len);
  }
};
