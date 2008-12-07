function SetTextBoxOutput(handles, hObject, method)
  lines = {};
  if strcmp(method, 'PCA') == 1
      lines{end + 1} = 'M. P. Holmes, A. G. Gray, and C. L. Isbell, Jr. QUIC-SVD: Fast SVD Using Cosine Trees.';
      lines{end + 1} = 'In Advances in Neural Information Processing Systems (NIPS) 22, 2009.';
  end;
  if strcmp(method, 'MVU') == 1
      lines{end + 1} = 'N. Vasiloglou, A. Gray, and D. Anderson. Scalable Semidefinite Manifold Learning. MLSP, 2008';
  end;
  if strcmp(method, 'EXACT') == 1
      lines{end + 1} = 'A. Gray and A. Moore. N-body problems in statistical learning.';
      lines{end + 1} = 'In NIPS, volume 14, pages 521-527, 2000';
  end;
  if strcmp(method, 'APPROX') == 1
      lines{end + 1} = 'Ram, P. and Lee, D. and Ouyang H. and Gray, A. Rank-Approximate Nearest Neighbors.';
      lines{end + 1} = 'To be submitted, 2009.';
  end;
  if strcmp(method, 'EMST') == 1
      lines{end + 1} = 'March, W. B. and Gray, A. Large-Scale Euclidean MST and Hierarchical Clustering. to be submitted, 2009.';
  end;
  if strcmp(method, 'KPCA') == 1
      lines{end + 1} = 'M.P. Holmes and A. Gray. QUIK-SVD: Fast Kernel PCA and Kernel Matrix Inversion Using Kernel Trees.';
      lines{end + 1} = 'To be submitted, 2009.';
  end;
  if strcmp(method, 'KDE') == 1
      lines{end + 1} = 'D. Lee and A. Gray. Faster gaussian summation: Theory and experiment.';
      lines{end + 1} = 'In Proceedings of the 22nd Annual Conference on Uncertainty in Artificial Intelligence (UAI-06),';
      lines{end + 1} = 'Arlington, Virginia, 2006.';
      lines{end + 1} = 'D. Lee and A. Gray. Fast High-dimensional Kernel Summatins Using the Monte Carlo Multipole Method.';
      lines{end + 1} = 'In Advances in Neural Information Processing Systems (NIPS) 22, 2009.';
      lines{end + 1} = 'D. Lee, A. Gray, and A. Moore. Dual-tree fast gauss transforms.';
      lines{end + 1} = 'Advances in Neural Information Processing Systems, pages 747-754.';
  end;
  if strcmp(method, 'Range search') == 1
  end;
  if strcmp(method, 'NBC') == 1
  end;
  set(handles.textbox_output, 'String', lines);
  guidata(hObject, handles);
  drawnow;