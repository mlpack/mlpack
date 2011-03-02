#include "io.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <boost/exception/all.hpp>
using namespace mlpack;

/* For clarity, we will alias boost's namespace */
namespace po = boost::program_options;

/* Declare the singleton variable's initial value, NULL */
IO* IO::singleton = 0;

/* Constructors, Destructors, Copy */
IO::IO() : desc("Allowed Options") , hierarchy("Allowed Options") {
	return;
}

IO::IO(std::string& optionsName) : desc(optionsName.c_str()), hierarchy(optionsName.c_str()) {
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
	
	std::string path = IO::getSingleton().manageHierarchy(identifier, parent, description);
	
	desc.add_options()
		(path.c_str(), description);
	return;
}

template<class T>
void IO::add(const char* identifier, const char* description, const char* parent) {
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
	try{ 
		po::store(po::parse_command_line(argc, line, desc), vmap);
	}catch(std::exception& ex) {
		std::cout << ex.what();
	}
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
	IO::getSingleton().hierarchy.print();
}


std::string IO::manageHierarchy(const char* id, const char* parent, const char* description) {
	std::string path(id), desc(description);
	
	if(parent != NULL) {
		std::string p(parent);
		//Lets sanity check string, remove superfluous '/' prefixes
		if(p.find_first_of("/") == 0)
			p = p.substr(1,p.length()-1);
		//Add necessary '/' suffixes to parent
		if(p.find_last_of("/") != p.length()-1)
			p = p+"/";
		path = p+id;
	}
	
	//Add the sanity checked string to the hierarchy
	if(desc.length() == 0)
		hierarchy.appendNode(path);
	else
		hierarchy.appendNode(path, desc);
	return path;
}

