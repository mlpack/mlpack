unknown_phase = -1;
M_G1_boundary_phase = 0;
G1_phase = 1;
S_phase = 2;
S_G2_phase = 3;
G2_M_phase = 4;

unknown_cluster = -1;
CLB2_cluster = 0;
CLN1_cluster = 1;
Histone_cluster = 2;
MAT_cluster = 3;
MCM_cluster = 4;
MET_cluster = 5;
SIC1_cluster = 6;
Y_cluster = 7;


data = zeros(82, 6178);
phases = zeros(1,6178);
clusters = zeros(1,6178);

fid = fopen('genes/combined_phase_cluster.txt');

% throw away first line
textscan(fid, '%s', 83, 'delimiter', '\t', 'emptyValue', -inf);

cur_row = [1];

row_num = 0;

while(length(cur_row) > 0)
  %  phasestring %
  phasestring = textscan(fid, '%s', 1, 'delimiter', '\t', 'emptyValue', ...
			 -inf);
  phasestring = char(phasestring{1});
  if strcmp(phasestring, 'm_g1_boundary') == 1
    phase = M_G1_boundary_phase;
  elseif strcmp(phasestring, 'g1') == 1
    phase = G1_phase;
  elseif strcmp(phasestring, 's') == 1
    phase = S_phase;
  elseif strcmp(phasestring, 's_g2') == 1
    phase = S_G2_phase;
  elseif strcmp(phasestring, 'g2_m') == 1
    phase = G2_M_phase;
  elseif strcmp(phasestring, 'unknown') == 1
    phase = unknown_phase;
  else
    disp(phasestring);
  end
  
  %  clusterstring %
  clusterstring = textscan(fid, '%s', 1, 'delimiter', '\t', 'emptyValue', ...
			   -inf);
  clusterstring = char(clusterstring{1});
  if strcmp(clusterstring, 'CLB2') == 1
    cluster = CLB2_cluster;
  elseif strcmp(clusterstring, 'CLN2') == 1
    cluster = CLN1_cluster;
  elseif strcmp(clusterstring, 'Histone') == 1
    cluster = Histone_cluster;
  elseif strcmp(clusterstring, 'MAT') == 1
    cluster = MAT_cluster;
  elseif strcmp(clusterstring, 'MCM') == 1
    cluster = MCM_cluster;
  elseif strcmp(clusterstring, 'MET') == 1
    cluster = MET_cluster;
  elseif strcmp(clusterstring, 'SIC1') == 1
    cluster = SIC1_cluster;
  elseif strcmp(clusterstring, 'Y') == 1
    cluster = Y_cluster;
  elseif strcmp(clusterstring, 'unknown') == 1
    cluster = unknown_cluster;
  else
    disp(clusterstring);
  end
  
  
  textscan(fid, '%s', 1, 'delimiter', '\t', 'emptyValue', -inf);
  cur_row = textscan(fid, '%f', 82, 'delimiter', '\t', ...
		     'emptyValue', -inf);
  cur_row = cur_row{1};
  
  if(length(cur_row) > 0)
    row_num = row_num + 1;
    while length(cur_row) < 82
      cur_row(end+1) = -inf;
    end
    data(:,row_num) = cur_row;
    phases(row_num) = phase;
    clusters(row_num) = cluster;
  end
end
