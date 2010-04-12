function saveHMM(fn, tr, e)
fid = fopen(fn, 'w');
fprintf(fid, '%% transition matrix ,\n');
fprintf(fid, formatStr(tr), tr);
fprintf(fid, '%% emission matrix ,\n');
fprintf(fid, formatStr(e), e);
fclose(fid);
end

function str = formatStr(m)
n = size(m,1);
str = '';
for i=1:n
    str = [str '%d,'];
end
str = [str '\n'];
end