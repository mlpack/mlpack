function DimReduction(datafile, method)
% datafile is without the csv extension

  if strcmp(method, 'PCA')==1
      exec=[fastlib_root ' ' datafile '.csv']
      system(exec);
      return;  
  end
  
  if strcmp(method, 'MVU')==1
     % do  pca on the mvu file to bring it down to 2
     % and output the result to input.csv
     return;  
 end
 error('Method not supported');