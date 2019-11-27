#ifndef MLPACK_BINDINGS_JAVA_PRINT_JAVA_HPP
#define MLPACK_BINDINGS_JAVA_PRINT_JAVA_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace java {

void PrintJava(const util::ProgramDoc& programInfo, 
    const std::string& fileName, const std::string& className);

}
}
}

#endif
