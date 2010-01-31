function [tr, e] = loadHMM(fn, M, N)
fid = fopen(fn, 'r');
fgets(fid);
tr = fscanMat(fid, M, M);
fgets(fid);
e = fscanMat(fid, M, N);
fclose(fid);
end

function m = fscanMat(fid, m, n)
[tmp_col, count] = fscanf(fid,'%f,\n', inf);
m = reshape(tmp_col, m, n);
end