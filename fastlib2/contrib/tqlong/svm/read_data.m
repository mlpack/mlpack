function [X, y] = read_data(fn, n)
f = fopen(fn, 'r');
X = []; y=[];
format = '%f';
for i=1:n
    format = [format ' %f:%f'];
end
while (~feof(f))
    line = fgetl(f);
    a = sscanf(line,format);
    if (length(a) ~= 9) error('error'); end
    X = [X; a(3:2:9)']; y = [y; a(1)];
end
