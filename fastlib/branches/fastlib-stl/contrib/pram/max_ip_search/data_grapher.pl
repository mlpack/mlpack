$prefix = $ARGV[0];

$file_list = `ls $prefix*.txt`;
@files = split /\n/, $file_list;

my @names;
my @speedups;
my $num_lines;

foreach $file(@files) {
    
    open IN, $file or die "Cant open $file\n";
    my @speedup;
    while(<IN>) {
	chomp($_);
	s/ //g;
	push(@speedup, $_);
    }

    close IN;
    $num_lines = $#speedup+1;

    push(@speedups, \@speedup);

    @a = split /\./, $file;
    $_ = $a[0];
    s/$prefix//;

    print "$_ $num_lines\n";
    push(@names, $_);
}

$outfile = "$prefix"."_table.txt";
open OUT, ">$outfile" or die "Cant open $outfile\n";
for ($i = 0; $i <= $#names; $i++) {
    print OUT "$names[$i]";
    if ($i == $#names) {
	print OUT "\n";
    } else {
	print OUT ",";
    }
}

for ($i = 0; $i < $num_lines; $i++) {
    for ($j = 0; $j <= $#names; $j++) {
	print OUT "$speedups[$j][$i]";
	if ($j == $#names) {
	    print OUT "\n";
	} else {
	    print OUT ",";
	}
    }
}

close OUT;

$rfile = "rtest.R";
open OUT, ">$rfile" or die "Cant open $rfile\n";
print OUT "library(ggplot2);\n";
print OUT "DF = read.csv('$outfile');\n";
print OUT "R = stack(list('Single-tree'=DF[,4], 'Single-tree-AP'=DF[,3], 'Dual-tree'=DF[,2], 'Dual-tree-AP'=DF[,1]));\n";
print OUT "R\$k = c(1:$num_lines);\n";
print OUT "names(R) = c(\"speedup\", \"Algorithm\", \"k\");\n";
print OUT "pdf('$prefix.pdf');\n";
print OUT "print(qplot(x=k, y=speedup, color=Algorithm, geom=\"line\", data=R));\n";
print OUT "dev.off()\n";

close OUT;

$to = `Rscript $rfile`;
$to = `rm -rf $rfile`;
$to = `rm -rf $outfile`;
