#ifndef SPECIAL_LA_H
#define SPECIAL_LA_H
#define FAILURE -1
#define SUCCESS 0


//This is a class of special linear algebra routines that are not
//provided by lapack

class special_la{
  

  //This function appends the matrix m1 to m2

   static index_t AppendMatrixInit(Matrix &m1, Matrix &m2, Matrix *m3){

    
    index_t num_rows1=m1.n_rows();
    index_t num_rows2=m2.n_rows();

    index_t num_cols1=m1.n_cols();
    index_t num_cols2=m2.n_cols();

    index_t total_num_cols=
      num_cols1+num_cols2;

    if(num_rows1!=num_rows2){

      return FAILURE;

    }
    m3->Init(num_rows1,total_num_cols);

    for(index_t i=0;i<num_rows1;i++){

      index_t j;
      for(j=0;j<num_cols1;j++){
	
	m3->set(i,j,m1.get(i,j));

      }

      for(index_t k=0;k<num_cols2;k++){

	m3->set(i,j+k,m2.get(i,k));
      }

    }

    return SUCCESS;
   }
   
   static void AppendMatrixWithZerosInit(Matrix &m1, 
					 index_t num_zeros, 
					 Matrix *m3){
     
     
     index_t num_rows1=m1.n_rows();
          
     index_t num_cols1=m1.n_cols();

     index_t total_num_cols=
       num_cols1+num_zeros;
     
     m3->Init(num_rows1,total_num_cols);
     
     for(index_t i=0;i<num_rows1;i++){
       
       index_t j;
       for(j=0;j<num_cols1;j++){
	 
	 m3->set(i,j,m1.get(i,j));
	 
       }

       for(index_t k=0;k<num_zeros;k++){
	 
	 m3->set(i,j+k,0);
       }
       
     }
   }
};



#endif
