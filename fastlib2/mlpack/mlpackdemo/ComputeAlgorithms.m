function ComputeAlgorithms(data_file, method)
  data_file
  method
  if strcmp(method, 'PCA')==1
      data_file
      method
      system('setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../../contrib/tqlong/quicsvd && ./quicsvd_test');
  end
  if strcmp(method, 'KDE')==1
      
  end
  if strcmp(method, 'Range search')==1
      
  end
  if strcmp(method, 'NBC')==1
      
  end
  if strcmp(method, 'Decision Trees')==1
      
  end

