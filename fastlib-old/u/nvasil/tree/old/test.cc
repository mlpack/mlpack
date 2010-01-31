

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_of_process);

  if (my_rank==0) {
    for(int32 i=0; i<num_of_process; i++) {
  	  SendFilePart( );
    }
  else {
  	ReceiveFile();
  }
  
  if (my_rank==0) {
  	//Open file and map it
  } else {
  	//Open file and map it
  }
  
  PivotParent();
  MPI_Reduce( MPI_MAXLOC);
  
  // So we can build a class that will do the recursion if it is 
  // rank zero otherwise they will do the junk work 
   
   