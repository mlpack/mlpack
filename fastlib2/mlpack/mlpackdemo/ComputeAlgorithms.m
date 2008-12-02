function ComputeAlgorithms(data_file, method)
  if strcmp(method, 'PCA')==1
    data_file = ['/net/hg200/dongryel/MLPACK_test_datasets/' data_file '.csv'];
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../../contrib/tqlong/quicsvd && ./quicsvd_main --A_in=' data_file];
    system(command);
  end
  if strcmp(method, 'KDE')==1
    data_file = ['/net/hg200/dongryel/MLPACK_test_datasets/' data_file '.csv'];
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../kde/ && ./kde_cv_bin --data=' data_file ' --kde/kernel=gaussian --kde/probability=0.8 --kde/relative_error=0.1'];
    system(command);
  end
  if strcmp(method, 'Range search')==1
      
  end
  if strcmp(method, 'NBC')==1
      
  end
  if strcmp(method, 'Decision Trees')==1
      
  end

