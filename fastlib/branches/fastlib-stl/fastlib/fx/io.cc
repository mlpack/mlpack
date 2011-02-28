#include "io.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <string>

using namespace mlpack;

/* For clarity, we will alias boost's namespace */
namespace po = boost::program_options;

/* Declare the singleton variable's initial value, NULL */
IO* IO::singleton = 0;

/* Constructors, Destructors, Copy */
IO::IO() : desc("Allowed Options") {
	return;
}

IO::IO(std::string& optionsName) : desc(optionsName.c_str()) {
	return;
}

IO::IO(const IO::IO& other) : desc(other.desc){
	return;
}

IO::~IO() {
	return;
}

/* Methods */
void IO::add(const char* identifier, const char* description, const char* parent) {
	//TODO: use singleton for state, wrap this up a parallel data structure 
	po::options_description& desc = IO::getSingleton().desc;
	
	if(parent == NULL)
		parent = rootPath;
	
	desc.add_options()
		(identifier, description);
	return;
}

template<class T>
void add(const char* identifier, const char* description, const char* parent) {
	//TODO: use singleton for state, wrap this up a parallel data structure 
	po::options_description& desc = IO::getSingleton().desc;

	if(parent == NULL)
		parent = rootPath;
	
	desc.add_options()
		(identifier, po::value<T>(), description);
	return;
}

//Returns the sole instance of this class
IO& IO::getSingleton() {
	if(!singleton)
		singleton = new IO();
	return *singleton;
}	
	

void IO::parseCommandLine(int argc, char** line) {
	po::variables_map& vmap = getSingleton().vmap;
	po::options_description& desc = getSingleton().desc;
	
	//Parse the command line, place the options & values into vmap
	po::store(po::parse_command_line(argc, line, desc), vmap);
	//Flush the buffer, make sure changes are propogated to vmap
	po::notify(vmap);	
}


void IO::parseStream(std::istream& stream) {
	po::variables_map& vmap = getSingleton().vmap;
	po::options_description& desc = getSingleton().desc;
	
	//Parse the stream, place options & values into vmap
	po::store(po::parse_config_file(stream, desc), vmap);
	//Flush the buffer, make s ure changes are propgated to vmap
	po::notify(vmap);
}

//Prints the current state, right now just for debugging purposes
void IO::print() {
	std::cout << getSingleton().desc << std::endl;
}





