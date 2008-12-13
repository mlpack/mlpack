
#ifndef TEST_PARTICLE_STAT_H
#define TEST_PARTICLE_STAT_H


class TestParticleStat{

 public:
  void Init(){
    Matrix test_matrix;
    test_matrix.Init(7, 4);
    for(int i = 0; i < 4; i++){
      test_matrix.set(3,i, 2.0);
      test_matrix.set(4,i, 0.0);
      test_matrix.set(5,i, 0.0); 
      test_matrix.set(6,i, 0.0);
      test_matrix.set(0,i, i%2);
      test_matrix.set(1,i, floor(i/2));
      test_matrix.set(2,i, 0.0);
    }
    test_system_ = new ParticleStat;
    test_system_->Init(test_matrix, 0, 4);
 
  }  

  void Destruct(){
    delete test_system_;
  }

  void TestInit(){
    Init();  
   // Check kinematic stats
    printf("Mass: %f \n", test_system_->mass_);
    printf("Center of Mass: %f %f %f \n", test_system_->centroid_[0],
	   test_system_->centroid_[1],  test_system_->centroid_[2]);   
    Destruct();
  }

  void TestForce(){
    Init();
     Matrix test_matrix;
    test_matrix.Init(7, 4);
    for(int i = 0; i < 4; i++){
      test_matrix.set(3,i, 2.0);
      test_matrix.set(4,i, 0.0);
      test_matrix.set(5,i, 0.0); 
      test_matrix.set(6,i, 0.0);
      test_matrix.set(0,i, i%2);
      test_matrix.set(1,i, floor(i/2));
      test_matrix.set(2,i, 0.0);
    }

    Matrix forces;
    forces.Init(2, 4);
    for(int i = 0; i < 4; i++){
      forces.set(0, i, 0.3);
      forces.set(1, i, 1.0);
    }
    test_system_->InternalForce_(&test_matrix, forces,0.1);
    for (int j = 0; j < 4; j++){
       printf("Velocity %d: %f %f %f \n", j, test_matrix.get(4,j),
	      test_matrix.get(5,j),test_matrix.get(6,j));
    }
    Destruct();
  }

 


  void TestAll(){
    TestInit();   
    TestForce();
  }

 private: 
   ParticleStat *test_system_;
    

};

#endif
