/**
 * @file backtrace.cpp
 * @author Grzegorz Krajewski
 *
 * Implementation of the Backtrace class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <sstream>

#ifdef HAS_BFD_DL
  #include <execinfo.h>
  #include <signal.h>
  #include <unistd.h>
  #include <cxxabi.h>

  // Some versions of libbfd require PACKAGE and PACKAGE_VERSION to be set in
  // order for the include to not fail.  For more information:
  // https://github.com/mlpack/mlpack/issues/574
  #ifndef PACKAGE
    #define PACKAGE
    #ifndef PACKAGE_VERSION
      #define PACKAGE_VERSION
      #include <bfd.h>
      #undef PACKAGE_VERSION
    #else
      #include <bfd.h>
    #endif
    #undef PACKAGE
  #else
    #ifndef PACKAGE_VERSION
      #define PACKAGE_VERSION
      #include <bfd.h>
      #undef PACKAGE_VERSION
    #else
      #include <bfd.h>
    #endif
  #endif
  #include <dlfcn.h>
#endif

#include "prefixedoutstream.hpp"
#include "backtrace.hpp"
#include "log.hpp"

// Easier to read Backtrace::DecodeAddress().
#ifdef HAS_BFD_DL
  #define TRACE_CONDITION_1 (!dladdr(trace[i], &addressHandler))
  #define FIND_LINE (bfd_find_nearest_line(abfd, text, syms, offset, &frame.file, &frame.function, &frame.line) && frame.file)
#endif

using namespace mlpack;

// Initialize Backtrace static inctances.
Backtrace::Frames Backtrace::frame;
std::vector<Backtrace::Frames> Backtrace::stack;

#ifdef HAS_BFD_DL
// Binary File Descriptor objects.
bfd* abfd = 0;          // Descriptor datastructure.
asymbol **syms = 0;     // Symbols datastructure.
asection *text = 0;     // Strings datastructure.
#endif

#ifdef HAS_BFD_DL
Backtrace::Backtrace(int maxDepth)
{
  frame.address = NULL;
  frame.function = "0";
  frame.file = "0";
  frame.line = 0;

  stack.clear();

  GetAddress(maxDepth);
}
#else
Backtrace::Backtrace()
{
  // Dummy constructor
}
#endif

#ifdef HAS_BFD_DL
void Backtrace::GetAddress(int maxDepth)
{
  void* trace[maxDepth];
  int stackDepth = backtrace(trace, maxDepth);

  // Skip first stack frame (points to Backtrace::Backtrace).
  for (int i = 1; i < stackDepth; i++)
  {
    Dl_info addressHandler;

    //No backtrace will be printed if no compile flags: -g -rdynamic
    if (TRACE_CONDITION_1)
    {
      return ;
    }

    frame.address = addressHandler.dli_saddr;

    DecodeAddress((long)frame.address);
  }
}

void Backtrace::DecodeAddress(long addr)
{
  // Check to see if there is anything to descript. If it doesn't, we'll
  // dump running program.
  if (!abfd)
  {
    char ename[1024];
    int l = readlink("/proc/self/exe", ename, sizeof(ename));
    if (l == -1)
    {
      perror("Failed to open executable!\n");
      return;
    }
    ename[l] = 0;

    bfd_init();

    abfd = bfd_openr(ename, 0);
    if (!abfd)
    {
      perror("bfd_openr failed: ");
      return;
    }

    bfd_check_format(abfd,bfd_object);

    unsigned storage_needed = bfd_get_symtab_upper_bound(abfd);
    syms = (asymbol **) malloc(storage_needed);

    text = bfd_get_section_by_name(abfd, ".text");
   }

  long offset = addr - text->vma;

  if (offset > 0)
  {
    if (FIND_LINE)
    {
      DemangleFunction();
      // Save retrieved information.
      stack.push_back(frame);
    }
  }
}

void Backtrace::DemangleFunction()
{
  int status;
  char* tmp = abi::__cxa_demangle(frame.function, 0, 0, &status);

  // If demangling is successful, reallocate 'frame.function' pointer to
  // demangled name. Else if 'status != 0', leave 'frame.function as it is.
  if (status == 0)
  {
    frame.function = tmp;
  }
}
#else
void Backtrace::GetAddress(int /* maxDepth */) { }
void Backtrace::DecodeAddress(long /* address */) { }
void Backtrace::DemangleFunction() { }
#endif

std::string Backtrace::ToString()
{
  std::string stackStr;

#ifdef HAS_BFD_DL
  std::ostringstream lineOss;
  std::ostringstream it;

  if (stack.size() <= 0)
  {
    stackStr = "Cannot give backtrace because program was compiled";
    stackStr += " without: -g -rdynamic\nFor a backtrace,";
    stackStr += " recompile with: -g -rdynamic.\n";

    return stackStr;
  }

  for (size_t i = 0; i < stack.size(); i++)
  {
    frame = stack[i];

    lineOss << frame.line;
    it << i + 1;

      stackStr += "[bt]: (" + it.str() + ") "
          + frame.file + ":"
          + lineOss.str() + " "
          + frame.function + ":\n";

    lineOss.str("");
    it.str("");
  }
#else
  stackStr = "[bt]: No backtrace for this OS. Work in progress.";
#endif

  return stackStr;
}
