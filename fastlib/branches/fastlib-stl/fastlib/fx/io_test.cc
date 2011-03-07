#include "io.h"
#include "optionshierarchy.h"
#include <string>

using namespace mlpack;

int main(int argc, char** argv) {
	IO::add("test", "description");
	IO::add("help", "This is a help message", "/bar/foo/");
	IO::add("foo", "durka durka", "bar");
	IO::add("help1", "This is a help message", "/bar/foo");
	IO::add("help2", "This is a help message", "barfoo");
	IO::add("help3", "This is a help message", "/bar/foo/");
	IO::add("bar", "This is a submodule");
	IO::add<std::string>("test3",  "This should be a string", NULL, true);
	
	IO::parseCommandLine(argc, argv);
	
	std::string x;
	if(IO::checkValue("test3")) {
		x = IO::getValue<std::string>("test3") ;
		std::cout << x << std::endl;
	}
	
}