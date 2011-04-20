/* Adds an option and global variable to IO */
template<typename T>
void IO::Add(const char* identifier, 
             const char* description, 
             const char* parent, 
             bool required) {

  po::options_description& desc = IO::GetSingleton().desc;
  //Generate the full pathname and insert the node into the hierarchy
  std::string tmp = TYPENAME(T);
  std::string path = IO::GetSingleton().ManageHierarchy(identifier, parent, 
    tmp, description);

  //Add the option to boost program_options
  desc.add_options()
    (path.c_str(), po::value<T>(),  description);
  //If the option is required, add it to the required options list
  if (required) 
    GetSingleton().requiredOptions.push_front(path);
  return;
}

/* Adds a complex type to the global values system, but creates no option */
template<class T>
void IO::AddComplexType(const char* identifier, 
                        const char* description,
                        const char* parent) {

  //Use singleton for state, wrap this up in a parallel data structure
  std::string type = TYPENAME(T);
            
  //Generate the full path string, and place the node in the hierarchy
  std::string path = GetSingleton().ManageHierarchy(identifier, parent, 
    type, description);
}

/* Returns a value of the specified type, creates one if not found */   
template<typename T>
T& IO::GetValue(const char* identifier) {
  //Used to ensure we have a valid value
  T tmp;
  //Used to index into the globalValues map
  std::string key = std::string(identifier);
  std::map<std::string, boost::any>& gmap = GetSingleton().globalValues;

  //If we have the option, set it's value
  if (CheckValue(identifier) && !gmap.count(key)) {
    gmap[key] = boost::any(GetSingleton().vmap[identifier].as<T>());
  }

  //We may have whatever is on the commandline, but what if
  //The programmer has made modifications?
  if (!gmap.count(key)) {//The programmer hasn't done anything, lets register it
    gmap[key] = boost::any(tmp);
    *boost::any_cast<T>(&gmap[key]) = tmp;
  }

  return *boost::any_cast<T>(&gmap[key]);
}


/* This class is used to facilitate easy addition of options to the program. 
*/
template<typename N>
class Option {
  //Base add an option
  public:
    Option(bool ignoreTemplate, 
           const char* identifier, 
           const char* description, 
           const char* parent=NULL, 
           bool required=false) {

      if(ignoreTemplate)
        IO::Add(identifier, description, parent, required);
      else
        IO::Add<N>(identifier, description, parent, required);

    }   
    
    Option(const char* identifier, 
           const char* description, 
           const char* parent=NULL, 
           bool required=false) {

      IO::AddComplexType<N>(identifier, description, parent);

  }   
};  

