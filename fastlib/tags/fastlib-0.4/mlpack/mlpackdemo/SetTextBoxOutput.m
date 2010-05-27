function SetTextBoxOutput(handles, hObject, method)
  lines = {};
  if strcmp(method, 'PCA') == 1
      lines{end + 1} = 'M. P. Holmes, A. G. Gray, and C. L. Isbell, Jr.';
      lines{end + 1} = 'QUIC-SVD: Fast SVD Using Cosine Trees.';
      lines{end + 1} = 'In Advances in Neural Information Processing Systems (NIPS) 22, 2009.';
  end;
  if strcmp(method, 'MVU') == 1
      lines{end + 1} = 'N. Vasiloglou, A. Gray, and D. Anderson.';
      lines{end + 1} = 'Scalable Semidefinite Manifold Learning. MLSP, 2008';
  end;
  if strcmp(method, 'EXACT') == 1
      lines{end + 1} = 'A. Gray and A. Moore.';
      lines{end + 1} = 'N-body problems in statistical learning.';
      lines{end + 1} = 'In NIPS, volume 14, pages 521-527, 2000';
  end;
  if strcmp(method, 'APPROX') == 1
      lines{end + 1} = 'Ram, P. and Lee, D. and Ouyang H. and Gray, A.';
      lines{end + 1} = 'Rank-Approximate Nearest Neighbors.';
      lines{end + 1} = 'To be submitted, 2009.';
      lines{end + 1} = ' ';
      lines{end + 1} = 'Comparable Timings between the 2 algorithms';
      lines{end + 1} = 'LayoutHistogram-rnd.csv';
      lines{end + 1} = 'Time            LSH Rank Error       ApproxNN Rank Error';
      lines{end + 1} = '~1                     355.5                          17.4';
      lines{end + 1} = '~2                       12.0                          3.1';
      lines{end + 1} = '~4                         1.04                          0';
      lines{end + 1} = '~13                       0.02                           -';
      lines{end + 1} = '~18                       0                                -';
      lines{end + 1} = ' ';
      lines{end + 1} = 'mnist10k_test.csv';
      lines{end + 1} = 'Time            LSH Rank Error       ApproxNN Rank Error';
      lines{end + 1} = '~1.5                     471.0                         150.2';
      lines{end + 1} = '~4                         30.65                          67.5';
      lines{end + 1} = '~20                       20.61                          12.4';
      lines{end + 1} = '~30                         3.2                             6.54';
      lines{end + 1} = '~210                       0                                0 (in ~190)';
  end;
  if strcmp(method, 'EMST') == 1
      lines{end + 1} = 'March, W. B. and Gray, A.';
      lines{end + 1} = 'Large-Scale Euclidean MST and Hierarchical Clustering. to be submitted, 2009.';
  end;
  if strcmp(method, 'KPCA') == 1
      lines{end + 1} = 'M.P. Holmes and A. Gray.';
      lines{end + 1} = 'QUIK-SVD: Fast Kernel PCA and Kernel Matrix Inversion Using Kernel Trees.';
      lines{end + 1} = 'To be submitted, 2009.';
  end;
  if strcmp(method, 'KDE') == 1
      lines{end + 1} = 'D. Lee and A. Gray.';
      lines{end + 1} = 'Faster gaussian summation: Theory and experiment.';
      lines{end + 1} = 'In Proceedings of the 22nd Annual Conference on Uncertainty in Artificial Intelligence (UAI-06),';
      lines{end + 1} = 'Arlington, Virginia, 2006.';
      lines{end + 1} = ' ';
      lines{end + 1} = 'D. Lee and A. Gray.';
      lines{end + 1} = 'Fast High-dimensional Kernel Summatins Using the Monte Carlo Multipole Method.';
      lines{end + 1} = 'In Advances in Neural Information Processing Systems (NIPS) 22, 2009.';
      lines{end + 1} = ' ';
      lines{end + 1} = 'M. P. Holmes, A.G. Gray, and C. L. Isbell, Jr.';
      lines{end + 1} = 'Ultrafast Monte Carlo for Kernel Estimators for Generalized Statistical Summations.';
      lines{end + 1} = 'In Advances in Neural Information Processing Systems (NIPS) 21, 2008.';
      lines{end + 1} = ' ';
      lines{end + 1} = 'D. Lee, A. Gray, and A. Moore.';
      lines{end + 1} = 'Dual-tree fast gauss transforms.';
      lines{end + 1} = 'Advances in Neural Information Processing Systems, pages 747-754.';
  end;
  if strcmp(method, 'Range search') == 1
  end;
  if strcmp(method, 'NBC') == 1
  end;
  set(handles.textbox_output, 'String', lines);
  guidata(hObject, handles);
  drawnow;