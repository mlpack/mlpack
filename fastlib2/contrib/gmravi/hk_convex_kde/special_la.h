#ifndef SPECIAL_LA_H
#define SPECIAL_LA_H
#define FAILURE -1
#define SUCCESS 0


//This is a class of special linear algebra routines that are not
//provided by lapack

class special_la{

 public:  
  
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

   static index_t StackMatrixMatrixInit(Matrix m1,Matrix m2,Matrix *m3){

     printf("Will stack matrices...\n");

    
     index_t num_cols1=m1.n_cols();
     index_t num_cols2=m2.n_cols();

     if(num_cols1!=num_cols2){

       printf("Coluns1=%d and Columns2=%d...\n",num_cols1,num_cols2);

       return FAILURE;
     }

     index_t num_rows1=m1.n_rows();
     index_t num_rows2=m2.n_rows();

     m3->Init(num_rows1+num_rows2,num_cols1);

     index_t i=0;
     for(i=0;i<num_rows1;i++){

       for(index_t j=0;j<num_cols1;j++){

	 m3->set(i,j,m1.get(i,j));
       }
     }
     for(index_t k=0;k<num_rows2;k++){
       for(index_t j=0;j<num_cols2;j++){
	 
	 m3->set(i+k,j,m2.get(k,j));
       }
     }
     return SUCCESS;
   }

 static index_t StackVectorVectorInit(Vector v1,Vector v2,Matrix *m3){

    
     index_t num_cols1=v1.length();
     index_t num_cols2=v2.length();

     if(num_cols1!=num_cols2){
       
       return FAILURE;
     }

     //Initialize m3
     m3->Init(2,num_cols1);
     for(index_t j=0;j<num_cols1;j++){
       
       m3->set(0,j,v1[j]);
     }
     
     for(index_t k=0;k<num_cols1;k++){
       
       m3->set(1,k,v2[k]);
     }
     return SUCCESS;
 }
 
 static index_t StackMatrixVectorInit(Matrix m1,Vector v2,Matrix *m3){
   
   printf("Came to matrix vector stacking...\n");
   index_t num_cols1=m1.n_cols();
   index_t num_cols2=v2.length();
   
   if(num_cols1!=num_cols2){
     
     printf("Cols1=%d and cols2=%d...\n",num_cols1,num_cols2);
     
     return FAILURE;
   }
   
   printf("Passed the test...\n");
   index_t num_rows1=m1.n_rows();
   
   m3->Init(num_rows1+1,num_cols1);
   
   index_t i=0;
   for(i=0;i<num_rows1;i++){
     
     for(index_t j=0;j<num_cols1;j++){
       
       m3->set(i,j,m1.get(i,j));
     }
   }

   for(index_t j=0;j<num_cols2;j++){
     
     m3->set(i,j,v2[j]);
   }
   return SUCCESS;
 }

 static index_t StackVectorMatrixInit(Vector v1,Matrix m1,Matrix *m2){
   
   
   index_t num_cols1=m1.n_cols();
   index_t num_cols2=v1.length();
   
   if(num_cols1!=num_cols2){
     
     printf("Cols1=%d and cols2=%d...\n",num_cols1,num_cols2);
     
     return FAILURE;
   }
   
   index_t num_rows1=m1.n_rows();
   m2->Init(num_rows1+1,num_cols1);
  

   index_t i=0;
   for(index_t j=0;j<num_cols2;j++){
     
     m2->set(i,j,v1[j]);
   }
   for(index_t i=1;i<=num_rows1;i++){
     
     for(index_t j=0;j<num_cols1;j++){
       
       m2->set(i,j,m1.get(i-1,j));
     }
   }

   return SUCCESS;
 }

};



#endif
