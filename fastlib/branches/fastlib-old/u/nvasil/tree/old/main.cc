#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include "boost/program_options.hpp"
#include "boost/type_traits.hpp"
#include "base/basic_types.h"
#include "smart_memory/src/memory_manager.h"
#include "data_file.h"
#include "data_reader.h"
#include "kdnode.h"
#include "euclidean_ball_node.h"
#include "tree.h"
#include "timit/transcript.h"
typedef float32 Precision_t;
typedef uint64  IdPrecision_t;
typedef MemoryManager<true> AllocatorWithLog_t;
typedef MemoryManager<false> AllocatorNoLog_t;

typedef Tree<Precision_t, IdPrecision_t, AllocatorWithLog_t, true,
				       KdNode> KdTreeWithLog_t;

typedef Tree<Precision_t, IdPrecision_t, AllocatorNoLog_t, true,
				       KdNode> KdTreeNoLog_t ;
typedef Tree<Precision_t, IdPrecision_t, AllocatorWithLog_t, true,
				     EuclideanBallNode> BallTreeWithLog_t;
typedef Tree<Precision_t, IdPrecision_t, AllocatorNoLog_t, true,
				     EuclideanBallNode> BallTreeNoLog_t;
struct Arguments {
	string Print();
  string neighbor_type;
	bool memory_log;
	string tree_struct;
  string tree_type;
  string build_type;
  int32 knns;
  float32 range;
  string data_file;
  string out_file;
  string log_file;
	string param_log_dir;
  string temp_folder;
  string mem_file;
  uint64 memory_capacity;
  float32 data_percent;
	bool build_tree;
	bool generate_random;
	uint64 num_of_points;
 	int32 dimension;
  PointIdentityDiscriminator<IdPrecision_t>::DiscriminantType 
		validator;
	string transcript_file;
};


namespace po = boost::program_options;
using namespace std;


template <typename NEIGHBORTYPE, typename ALLOCATOR, typename TREETYPE>
void MakeGraphSingle(TREETYPE &tree,
                     NEIGHBORTYPE range,
                     IdPrecision_t num_of_points,
                     DataReader<Precision_t, IdPrecision_t> *data,
                     string filename);

template <typename NEIGHBORTYPE, typename TREETYPE>
void MakeGraphDual(TREETYPE &tree,
                   NEIGHBORTYPE range,
                   string filename);                     
void MakeGraphNaive(DataReader<Precision_t, IdPrecision_t> &data, 
		                IdPrecision_t num_of_points,
										int32 dimension,
										int32 knns,
										string filename);
template<typename ALLOCATOR, typename TREETYPE>
int32 CoreOperations(Arguments &args, 
		                DataReader<Precision_t, IdPrecision_t> *data,
										int32 dimension,
										IdPrecision_t num_of_points);
void GenerateRandom(Arguments &arg);
FILE *out_log;

int main(int argc, char *argv[]) {
    Arguments args;
		int temp_validator;
  try {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
				("memory_log", po::value<bool>(&args.memory_log)->default_value(false))
				("tree_struct", po::value<string>(&args.tree_struct)->default_value("kd"),
				 "tree_struct,\n"
				 "kd: kdtree\n"
				 "ball: euclidean ball tree\n"
				 "default: kd\n")
        ("tree_type", po::value<string>(&args.tree_type)->default_value("single"),
         "tree type,\n" 
         "single: single tree\n"
         "dual: dual tree\n"
				 "naive: brute force, no tree is used\n"
         "default is single\n")
        ("build_type", po::value<string>(&args.build_type)->default_value("depth"),
         "build method,\n" 
         "depth: depth first\n"
         "breadth: breadth\n"
         "default is depth\n")
        ("neighbor_type", po::value<string>(&args.neighbor_type)->default_value("k"),
         "type of neighbors,\n" 
         "k: k nearest\n"
         "range: range nearest\n"
         "default is k\n")
        ("k", po::value<int>(&args.knns)->default_value(5), 
         " number of k nearest neighbors, default is 5\n")
        ("data_percent",  po::value<float32>(&args.data_percent)->default_value(1),
         "If you don't want to use all the points of the dataset specify "
         "data_percent between (0,1] to choose how many points you want "
         "to use. The default value is 1\n")
        ("range", po::value<float32>(&args.range)->default_value(0.2),
         "range for range-neighbors search, default is 0.2\n")
        ("input_file", po::value<string>(&args.data_file),
         "Input data file that contains the data, It must "
         "have a specific format\n")
        ("output_file", po::value<string>(&args.out_file)->default_value("allnn"),
         "output file in ASCII format for range search. First column is the point id,"
         " Second Column is the neighbor point id and the third "
         "is the squared distance. Default value allnn\n"
				 "For k-nearest_neighbor search it is in binary format\n"
				 "point_id->uint64,\n"
				 "dummy->uint64\n"
				 "nearest_point_id->uint64\n"
				 "distance->float32\n") 
        ("log_file", po::value<string>(&args.log_file)->default_value("/dev/null"),
         "log file, during the creation of the tree, default value /dev/null\n")          
        ("param_log_dir", po::value<string>(&args.param_log_dir)->default_value(""),
         "directory where the parameters of the execution are stored\n")          
				("temp_folder", po::value<string>(&args.temp_folder)->default_value("./temp/"),
         "this is the temporary folder for storing temp files. Default value is "
         " ./temp/ (the program creates this folder automatically)\n")
        ("mem_file", po::value<string>(&args.mem_file)->
         default_value("./memorymanager"),
         "This is the file name of the memory manager that contains the tree "
         "You can use this file to reload the tree. Default value "
         "./temp/memorymanager\n")
        ("capacity", po::value<uint64>(&args.memory_capacity)->default_value(65536*1024), 
         "The capacity for the memory manager (in bytes), "
         "If you are loading a memory "
         "manager file this value is ignored. "
         "The default value is 67108864 bytes, (64MB)\n")
        ("build_tree",po::value<bool>(&args.build_tree)->default_value(true),
         "If you have already built the tree and have it saved in a " 
         "memory manager file set it false. The program will load the memory "
         "manager file and use this tree. Otherwise it will build it from the " 
         "data. The default value is true\n")
        ("generate_random", po::value<bool>(&args.generate_random)->default_value(false),
				 "flag to generate random synthetic data")
				("dimension", po::value<int32>(&args.dimension)->default_value(3),
			   "dimension of synthetic data")
				("num_of_points", po::value<uint64>(&args.num_of_points)->default_value(1000000))
				("validator", po::value<int>(&temp_validator)->default_value(0),
				 "validator: if 0 then it will consider points that have different id\n"
				 "if 1 then it considers points that are from different speaker\n")
				("transcript_file", po::value<string>(&args.transcript_file),
				 "transcript_file: contains the timit information, used for LOOCV flags ")
        ;

    po::variables_map vm;        
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help") || argc == 1) {
      printf(" This program intends to build a proximity graph of points "
      "with the intention to use it for manifold learning. It structures "
      "the data on a kd-tree and then runs all (range or k) nearest neighbors "
      "then outputs the proximity information on a text file with the following "
      "format:\n\n"
      "point_id   point_id  squared distance (||x_i - x_j||^2) \n"
      "You can easily import the data in Matlab with the command: \n"
      "[i j k] = textread(filename,'%%i%%i%%f') \n"
      "If you want to create the adjacency matrix of the graph just type:\n"
      "A = sparse(i,j,k)\n"
      "\n\n"
      "Comment: this program makes use of the memory manager which is a "
      "cache that can be used to build trees that don't fit in the RAM. "
      "Very soon we will have a version that will be able to use more than "
      "one computers" 
      "\n\n"
      "Usage:\n");
      cout << desc << "\n";
      printf("Developped by Nikolaos Vasiloglou II for the FastLAIb Project\n");
      fflush(stdout);
      return 1;
    } else {
      // Output the values set before the program starts running
      //cout << desc << "\n";   
    }
  }
  catch(exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
  }
	args.validator=PointIdentityDiscriminator<IdPrecision_t>::DiscriminantType 
      (temp_validator);
  time_t time1=time(NULL);
  struct tm today;
  gmtime_r(&time1, &today);
	char today_str[4096];
	strftime(today_str, 4096, "%Y_%b_%a_%d_%H_%M_%S", &today);
	string out_log_file(args.param_log_dir);
	out_log_file.append(today_str);
	out_log_file.append(".log");
  out_log = fopen(out_log_file.c_str(), "w");
  // append the log file to the allnn file so that you know 
	// the parameters
	args.out_file.append(today_str);

	if (out_log==NULL) {
	  fprintf(stderr, "Cannot create file: %s\n", out_log_file.c_str());
		assert(false);
	}
	fprintf(out_log, args.Print().c_str());

	if (args.generate_random) {
		fprintf(out_log,"Generating random data....\n");
		fprintf(out_log,"number of data :%llu\n", (unsigned long long)args.num_of_points);
		fprintf(out_log,"dimension      :%d\n", args.dimension);
		fflush(out_log);
	  GenerateRandom(args);
		printf("\n");
	}
  IdPrecision_t map_size;
  int32  dimension;
  IdPrecision_t num_of_points;
  DataReader<Precision_t, IdPrecision_t> *data;
  if (args.data_file.empty() && args.build_tree==true) {
  	fprintf(out_log,"data_file option not specified, aborting....\n");
		fflush(out_log);
  	abort();
  } else {
  	void *temp;
  	OpenDataFile(args.data_file, &dimension, 
                 &num_of_points, 
                 &temp,
                 &map_size);
     data = new DataReader<Precision_t, IdPrecision_t>(temp, dimension);
		 num_of_points=(IdPrecision_t)(args.data_percent *num_of_points);
		 args.num_of_points=num_of_points;
  }
  // create the temp folder
  fprintf(out_log, "Creating temp folder %s\n", args.temp_folder.c_str());
	fflush(out_log);
  int return_code = mkdir(args.temp_folder.c_str(), 0755);
  if (return_code == -1 && errno!=17) {
  	fprintf(out_log, "mkdir(%s) returned %d: errno = %d [%s]\n",
  	       args.temp_folder.c_str(),return_code, errno,
  	       strerror(errno));
  	return -1;
  }

	if (args.tree_type=="naive") {
	  if (args.neighbor_type!="k") {
		  fprintf(stderr, "Naive method supported only for k nearest\n");
      return_code=-1;
			goto termination;
		}
	  fprintf(out_log, "Now computing all nearest with the naive method\n");
		fflush(out_log);
		time_t t1=time(NULL);
		MakeGraphNaive(*data, num_of_points, dimension, args.knns, args.out_file);
		time_t t2=time(NULL);
    fprintf(out_log,"Graph completed, using all-%d nearest neighbors\n", args.knns);
    fprintf(out_log,"time elapsed: %lg seconds\n", (double)difftime(t2, t1));
		fflush(out_log);
		return_code=1;
		goto termination;
	}
  
	if (args.memory_log==true){
		if (args.tree_struct=="kd"){
		  return_code = CoreOperations<AllocatorWithLog_t, KdTreeWithLog_t> 
				               (args, data, dimension, num_of_points);
		} else {
		  if (args.tree_struct=="ball"){
	      return_code = CoreOperations<AllocatorWithLog_t, BallTreeWithLog_t> 
				               (args, data, dimension, num_of_points);
			} else {
			  fprintf(out_log, "This type of tree %s has not been implemented yet\n",
		    args.tree_struct.c_str());
				fflush(out_log);
		  	return_code=-1;
			  goto termination;
		  } 
	  }
	} else {
	  if (args.tree_struct=="kd"){
		  return_code = CoreOperations<AllocatorNoLog_t, KdTreeNoLog_t> 
				               (args, data, dimension, num_of_points);

		} else {
		  if (args.tree_struct=="ball"){
			return_code = CoreOperations<AllocatorNoLog_t, BallTreeNoLog_t> 
				               (args, data, dimension, num_of_points);

			} else {
		    fprintf(out_log, "This type of tree %s has not been implemented yet\n",
				  	   args.tree_struct.c_str());
				fflush(out_log);
			  return_code=-1;
			  goto termination;
		  }
	 }
	}

  
  // in case something goes wrong all the errors should jump here
 // for closing all the files and terminating
 termination:
  // remove the temp folder
  fprintf(out_log, "Removing temp folder %s\n", args.temp_folder.c_str());
  fflush(out_log);
  string temp =args.temp_folder;
  temp.append("*");
  unlink(temp.c_str());
   
  if (rmdir(args.temp_folder.c_str()) == -1) {
  	fprintf(out_log, "rmdir(%s) returned %d: errno = %d [%s]\n",
  	       args.temp_folder.c_str(),return_code, errno,
  	       strerror(errno));
		fflush(out_log);
  	return_code=-1;       
  }
  // Close the data file,
	delete data;
  CloseDataFile(data, map_size);
	fclose(out_log);
  return return_code;
}

template<typename ALLOCATOR, typename TREETYPE>
int32 CoreOperations(Arguments &args, 
		                DataReader<Precision_t, IdPrecision_t> *data,
										int32 dimension,
										IdPrecision_t num_of_points) {
  ALLOCATOR::allocator =  new ALLOCATOR();
	ALLOCATOR::allocator->set_pool_name(args.mem_file);
	ALLOCATOR::allocator->set_capacity(args.memory_capacity);
	ALLOCATOR::allocator->set_log_file(args.log_file);
  ALLOCATOR::allocator->Initialize();
  TREETYPE tree(data, dimension, IdPrecision_t(num_of_points));
  PointIdentityDiscriminator<IdPrecision_t> *discriminator;
	Transcript *transcript=NULL;
	if (args.validator==0) {
    discriminator = new PointIdentityDiscriminator<IdPrecision_t>();
	}	else {
		transcript = OpenBinaryTranscriptFile(args.transcript_file);
	  discriminator = new PointIdentityDiscriminator<IdPrecision_t>(
				args.validator, transcript);
		tree.set_discriminator(discriminator);
		printf("Timit discriminator chosen %i\n", args.validator);

	}
  int32 return_code;
	time_t t1,t2;
  if (args.build_tree == true) {
  	t1=time(NULL);
  	fprintf(out_log, "Building the tree...\n");
		fflush(out_log);
  	if (args.build_type==string("depth")) {
  	  tree.SerialBuildDepthFirst();
  	} else {
  		if  (args.build_type==string("breadth")) {
  	      tree.SerialBuildBreadthFirst();
  		} else {
  	      fprintf(out_log, "This option is not implemented yet,\n");
					fflush(out_log);
  	      return_code = -1;
  	      goto local_termination;
  	    }
  	}
  } else {
  	fprintf(out_log, "Initialization of the tree from a memory manager file "
  	       "not implemented yet\n");
		fflush(out_log);
  	return_code = -1;
  	goto local_termination;
  }
  t2=time(NULL);
  fprintf(out_log, "Tree built... in %lg seconds\n",(double)difftime(t2, t1));
  fprintf(out_log, "%s", tree.Statistics().c_str());
  fprintf(out_log, "Memory usage :%llu\n", (unsigned long long)
			                           ALLOCATOR::allocator->get_usage()); 
	fflush(out_log);
  ALLOCATOR::allocator->set_log(true);
  // reset counters so that we can see how much time is spent on the tree
  tree.ResetCounters();
  t1=time(NULL);
  if (args.tree_type==string("single")) {
  	if (args.neighbor_type==string("k") ){
      MakeGraphSingle<int32, ALLOCATOR, TREETYPE>(tree, args.knns,
                             IdPrecision_t(num_of_points ),
                             data,
                             args.out_file); 
    } else {
      if (args.neighbor_type==string("range")) {
        MakeGraphSingle<Precision_t, ALLOCATOR, TREETYPE>(tree, args.range,
                                     IdPrecision_t(num_of_points),
                                     data,
                                     args.out_file); 
      	
      } else {
      	fprintf(out_log, 
				    "This method %s is not implemented yet\n", args.neighbor_type.c_str());
				fflush(out_log);
      	return_code=-1;
      	goto local_termination;
      }
    }
  } else {
    if (args.tree_type==string("dual")) {
      if (args.neighbor_type == string("k")){
      	MakeGraphDual<int32, TREETYPE>(tree,
                                       args.knns,
                                       args.out_file);
      }	else {
      	if (args.tree_type==string("range")) {
      	  MakeGraphDual<Precision_t, TREETYPE>(tree,
                                               args.range,
                                               args.out_file);
      	} else {
      	  fprintf(out_log, "This method is not implemented yet\n");
					fflush(out_log);
      	  return_code=-1;
      	  goto local_termination;
        }
      }
    }
  }
  t2=time(NULL);
  return_code =1;
  fprintf(out_log,"Graph completed, using all-%d nearest neighbors\n", args.knns);
  fprintf(out_log,"time elapsed: %lg seconds\n", (double)difftime(t2, t1));
	fflush(out_log);
	if (transcript!=NULL) {
    CloseBinaryTranscriptFile(transcript, args.transcript_file);
  }

  /*
  printf("Total number of comparisons in the tree %llu\n",
         (unsigned long long int)
         tree.get_number_of_comparisons());
  printf("Total number of distances calculated %llu\n",
         (unsigned long long int)tree.get_distances_computed());
				 */
local_termination:
	delete ALLOCATOR::allocator;
	return return_code;
}

template <typename NEIGHBORTYPE, typename ALLOCATOR, typename TREETYPE>
void MakeGraphSingle(TREETYPE &tree,
                     NEIGHBORTYPE range,
                     IdPrecision_t num_of_points,
                     DataReader<Precision_t, IdPrecision_t> *data,
                     string filename) {
  fprintf(out_log,"Now computing the all nearest neighbors with the single tree\n");
	fflush(out_log);
	struct stat info1;
	typename TREETYPE::Result_t *out;
  FILE *fp;
	IdPrecision_t idx=0;

  if ((stat(filename.c_str(), &info1) != 0 || info1.st_size / 
			sizeof(typename TREETYPE::Result_t) !=num_of_points) &&
			boost::is_integral<NEIGHBORTYPE>::value) {
		time_t t1=time(NULL);
	  fp =fopen(filename.c_str(), "w");
    const int32 kChunk=8192;
		typename TREETYPE::Result_t buffer[kChunk*(int32)range];
		for(IdPrecision_t i=0; i<num_of_points/kChunk; i++) {
		  fwrite(buffer, sizeof(typename TREETYPE::Result_t), kChunk*(int32)range,
				 	fp);
		}
    fwrite(buffer, sizeof(typename TREETYPE::Result_t), 
				(num_of_points%kChunk)*(int32)range, fp);
    fclose(fp);
		time_t t2=time(NULL);
		fprintf(out_log, "Wasted %llu seconds to create the file\n", (unsigned long long)
				                                               t2-t1);
		fflush(out_log);
	} 
	if (boost::is_integral<NEIGHBORTYPE>::value) {
		int fd=open(filename.c_str(), O_RDWR);
	  out=(typename TREETYPE::Result_t *)mmap(NULL, num_of_points*(int32)range*
	      sizeof(typename TREETYPE::Result_t), PROT_WRITE | PROT_READ,
			  MAP_SHARED, fd, 0);
		if (out==MAP_FAILED) {
		  fprintf(out_log, "Failed to map the output: %s\n", strerror(errno));
			fflush(out_log);
			assert(false);
		}
	} else {
    fp=fopen(filename.c_str(), "w");	
		if (fp == NULL) {
  	  fprintf(out_log,"Could not open %s for writing the nearest neighbor info\n",
  	         filename.c_str());
			fflush(out_log);
  	  assert(false);
  	}
  
	}
	ShowProgress progress;
  progress.Reset();	
  for(IdPrecision_t i=0; i<num_of_points; i++) {

    vector<pair<Precision_t, Point<Precision_t, IdPrecision_t, ALLOCATOR> > > nearest;
  	Precision_t mindist;
  	Precision_t *test_point = data->At(i); 
    tree.NearestNeighbor(test_point,
                         &nearest,
                         &mindist,
                         range); 

    progress.Show(i, num_of_points);  
    if (boost::is_integral<NEIGHBORTYPE>::value) {
		  for(int32 j=0; j<range; j++) {
		    out[idx].point_id_=data->GetId(i);
		    out[idx].nearest_.set_id(nearest[j].second.get_id());	 
			  out[idx].distance_=nearest[j].first;
			  idx++;
		  }
	  } else	{
      for(IdPrecision_t j=0; j<nearest.size(); j++) {                              
        fprintf(fp, "%llu %llu %lg\n",
       	       (unsigned long long)data->GetId(i),
      	       (unsigned long long)nearest[j].second.get_id(),
      	       (double)nearest[j].first);
      }
    }
	}
  if (boost::is_integral<NEIGHBORTYPE>::value) {
	  munmap(out, num_of_points*(int32)range*
				sizeof(typename TREETYPE::Result_t));
	} else {
	  fclose(fp);
	}
  fprintf(out_log,"%s",tree.Computations().c_str());
	fflush(out_log);
} 

template <typename NEIGHBORTYPE, typename TREETYPE>
void MakeGraphDual(TREETYPE &tree,
                   NEIGHBORTYPE range,
                   string filename) {
  fprintf(out_log, "Now computing  all nearest neighbors with the dual tree\n");
	fflush(out_log);
  if ( !boost::is_floating_point<NEIGHBORTYPE>::value) {
	  time_t t1=time(NULL);
	  tree.InitAllKNearestNeighborOutput(filename, (int32)range);
	  time_t t2=time(NULL);
     fprintf(out_log, "Wasted %lu sec for creating the output file\n", t2-t1);
		 fflush(out_log);
	}
	tree.AllNearestNeighbors(tree.get_parent(), tree.get_parent(), (int32)range);
  if (!boost::is_floating_point<NEIGHBORTYPE>::value) {
  	tree.CloseAllKNearestNeighborOutput((int32)range);
	}
	fprintf(out_log, "\n");
	fprintf(out_log,"%s",tree.Computations().c_str());
	fflush(out_log);
}
void MakeGraphNaive(DataReader<Precision_t, IdPrecision_t> &data, 
		                IdPrecision_t num_of_points, 
										int32 dimension,
	                 	int32 knns, string filename) {
  fprintf(out_log, "Now computing all nearest neighbors with "
			             "the naive method\n");
	fflush(out_log);
  Precision_t *distances =(Precision_t *)mmap(NULL, 
			                         num_of_points * sizeof(Precision_t),
							                 PROT_WRITE | PROT_READ,
															MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  ShowProgress progress;
  progress.Reset();	
	if (distances==MAP_FAILED) {
	  fprintf(stderr, "Unable to map memory for naive distance computation:"
				            " %s\n", strerror(errno));
		assert(false);
	}
	for(IdPrecision_t i=0; i<num_of_points; i++) {
	  for(IdPrecision_t j=0; j<num_of_points; j++) {
		  distances[j]=0;
			for(int32 k=0; k<dimension; k++) {
			  distances[j]+=(data.At(i)[k]-data.At(j)[k])*
					             (data.At(i)[k]-data.At(j)[k]);
			} 
		}  
    for(int32 k=0; k<knns; k++) {
		  Precision_t min_dist=numeric_limits<Precision_t>::max();
		  IdPrecision_t min_ind=0;
		  for(IdPrecision_t j=0; j<num_of_points; j++) {
			  if (i!=j) {
			    if (min_dist> distances[j]){
				    min_dist=distances[j];
					  min_ind=j;
				  }
			  }
		  }
			distances[min_ind]=numeric_limits<Precision_t>::max();
		}
    progress.Show(i, num_of_points);  

	}

	munmap(distances, num_of_points *sizeof(Precision_t));
}

void GenerateRandom(Arguments &args) {
  ShowProgress progress;
  progress.Reset();	
	FILE *fp;
  fp=fopen(args.data_file.c_str(), "w");
	fwrite(&args.num_of_points, sizeof(uint64), 1, fp);
	fwrite(&args.dimension, sizeof(int32), 1, fp);
	uint64 total_bytes = args.num_of_points*(
			args.dimension*sizeof(float32)+sizeof(uint64));
	fwrite(&total_bytes, sizeof(uint64),1, fp);
	char buff[65536-20];
	fwrite(buff, 1, 65536-20, fp);
  for(uint64 i=0; i<args.num_of_points; i++) {
	  for(int32 j=0; j<args.dimension; j++) {
   	  float32 number = 1.0*rand()/RAND_MAX;
		  fwrite(&number, sizeof(float32), 1, fp);
	  }
    fwrite(&i, sizeof(uint64), 1, fp);
    progress.Show(i, args.num_of_points);
	}
  fclose(fp);
}

string Arguments::Print() {
  char buff[65536];
	sprintf(buff,"neighbor_type:   %s\n"
               "memory_log:      %i\n"
	             "tree_struct:     %s\n"
               "tree_type:       %s\n"
               "build_type:      %s\n"
               "knns:            %i\n"
							 "range:           %g\n"
               "data_file:       %s\n"
               "out_file:        %s\n"  
               "log_file:        %s\n"
               "temp_folder:     %s\n"
               "mem_file:        %s\n"
               "memory_capacity: %llu\n"
               "data_percent:    %g\n"
	             "build_tree:      %i\n"
	             "generate_random: %i\n"
	             "num_of_points:   %llu\n"
 	             "dimension:       %i\n",
                neighbor_type.c_str(),
	              memory_log,
	              tree_struct.c_str(),
                tree_type.c_str(),
                build_type.c_str(),
                knns,
                range,
                data_file.c_str(),
                out_file.c_str(),
                log_file.c_str(),
                temp_folder.c_str(),
                mem_file.c_str(),
               (unsigned long long)memory_capacity,
                data_percent,
	              build_tree,
	              generate_random,
	              (unsigned long long)num_of_points,
 	              dimension         
         );
  return string(buff);
}
