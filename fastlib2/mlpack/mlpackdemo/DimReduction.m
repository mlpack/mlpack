function DimReduction(datafile, method)
% datafile is without the csv extension

  if strcmp(method, 'PCA')==1
      exec=[fastlib_root ' ' datafile '.csv']
      system(exec);
      return;  
  end
  
  if strcmp(method, 'PCA')==1
    
     return;  
 end
 error('Method not supported');