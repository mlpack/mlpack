/***
 * @file option.cc
 * @author Ryan Curtin
 * 
 * Implementation of the ProgramDoc class.  The class registers itself with IO
 * when constructed.
 */
#include "io.h"
#include "option.h"

#include <string>

using namespace mlpack;
using namespace std;

/***
 * Construct a ProgramDoc object.  When constructed, it will register itself
 * with IO.  A fatal error will be thrown if more than one is constructed.
 *
 * @param programName Short string representing the name of the program.
 * @param documentation Long string containing documentation on how to use the
 *    program and what it is.  No newline characters are necessary; this is
 *    taken care of by IO later.
 */
ProgramDoc::ProgramDoc(std::string programName, std::string documentation) :
    programName(programName), documentation(documentation) {
  // Register this with IO.
  IO::RegisterProgramDoc(this);
}
