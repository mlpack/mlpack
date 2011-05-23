#ifndef MLPACK_IO_PRINTING_IMPL_H
#define MLPACK_IO_PRINTING_IMPL_H


template<typename T>
class PrefixedHelper {
  public:
   bool print(std::ostream& out, const T& rhs) {
     out << rhs;
     return false;
   }
  
   bool print(std::ostream& out, const char* prefix, const T& rhs) {
     out << prefix << rhs;
     return false;
   }
};

template<>
class PrefixedHelper<const char*> {
  public:
   bool print(std::ostream& out, const char* rhs) {
      out << rhs;
      int t = '\n';
      if(strchr(rhs, '\0') != NULL || strchr(rhs, '\n') != NULL)
        return true;
      else
        return false;
    }
    
    bool print(std::ostream& out, const char* prefix, const char* rhs) {
      out << prefix << rhs;
      
       if(strchr(rhs, '\0') != NULL || strchr(rhs, '\n') != NULL)
        return true;
      else
        return false;
    }
};

template<>
class PrefixedHelper<const std::string&> {
  public:
   bool print(std::ostream& out, const std::string& rhs) {
      out << rhs;
     
      if(rhs.find('\n') != std::string::npos || rhs.find('\0') != std::string::npos)
        return true;
      else
        return false;
    }
    
    bool print(std::ostream& out, const char* prefix, const std::string& rhs) {
      out << prefix << rhs;
      
      if(rhs.find('\n') != std::string::npos || rhs.find('\0') != std::string::npos)
        return true;
      else
        return false;
    }
};

template<typename T>
std::ostream& PrefixedOutStream::operator<<(const T& rhs){
  PrefixedHelper<T> helper;
  if (true)
    cariageReturned = helper.print(destination, prefix, rhs);
  else
    cariageReturned = helper.print(destination, rhs);

  return destination;
}

#endif //MLPACK_IO_PRINTING_IMPL_H