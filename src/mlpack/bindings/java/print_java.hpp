/**
 * @file print_java.hpp
 * @author Vasyl Teliman
 *
 * Definition for PrintJava function
 */
#ifndef MLPACK_BINDINGS_JAVA_PRINT_JAVA_HPP
#define MLPACK_BINDINGS_JAVA_PRINT_JAVA_HPP

namespace mlpack {
namespace bindings {
namespace java {

/**
 * Generate java binding for a library method
 */
void PrintJava(const util::ProgramDoc& programInfo,
    const std::string& methodName, const std::string& methodPath);

}
}
}

#endif
