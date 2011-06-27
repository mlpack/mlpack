if ($#ARGV != 6 && $#ARGV != 7) {
    die "Not the right number of arguments";
} else {

    if ($#ARGV == 6) {

	#$file_prefix = $ARGV[0];
	#$to = `ls $file_prefix*.csv`;
	#$num_classes = $ARGV[1];


	$train0 = $ARGV[0];
	$test0 = $ARGV[1];
	$train1 = $ARGV[2];
	$test1 = $ARGV[3];
	$folds = $ARGV[4];
	$min_leaf_size = $ARGV[5];
	$max_leaf_size = $ARGV[6];

	# computing the set sizes to compute the priors
	my @sizes;
	for ($i = 0; $i < 4; $i++) {
	    $to = `wc -l $ARGV[$i]`;
	    @a = split / /, $to;
	    $sizes[$i] = $a[0];
	    undef @a;
	}

	print "$sizes[0] - $sizes[1], $sizes[2] - $sizes[3]\n";


	$p0 = $sizes[0] / ($sizes[0] + $sizes[2]);
	$p1 = $sizes[2] / ($sizes[0] + $sizes[2]);


	$dt_out_f = "dt_out_tmp.csv";

	# computing 0on0
	$to = `./dt_driver --data=$train0 --folds=$folds --test=$test0 --test_output=$dt_out_f --dtree/min_leaf_size=$min_leaf_size --dtree/max_leaf_size=$max_leaf_size`;

	my @dt_00;
	open IN, $dt_out_f or die "Cant open $dt_out_f...\n";

	while(<IN>) {
	    push(@dt_00, $_);
	}

	close IN;
	$to = `rm -rf $dt_out_f`;

	# computing 0on1
	$to = `./dt_driver --data=$train1 --folds=$folds --test=$test0 --test_output=$dt_out_f --dtree/min_leaf_size=$min_leaf_size --dtree/max_leaf_size=$max_leaf_size`;

	my @dt_01;
	open IN, $dt_out_f or die "Cant open $dt_out_f...\n";

	while(<IN>) {
	    push(@dt_01, $_);
	}

	close IN;
	$to = `rm -rf $dt_out_f`;


	if ($#dt_00 != $#dt_01) {
	    die "00 = $#dt_00, 01 = $#dt_01...\n";
	}


	# computing 1on0
	$to = `./dt_driver --data=$train0 --folds=$folds --test=$test1 --test_output=$dt_out_f --dtree/min_leaf_size=$min_leaf_size --dtree/max_leaf_size=$max_leaf_size`;

	my @dt_10;
	open IN, $dt_out_f or die "Cant open $dt_out_f...\n";

	while(<IN>) {
	    push(@dt_10, $_);
	}

	close IN;
	$to = `rm -rf $dt_out_f`;

	# computing 1on1
	$to = `./dt_driver --data=$train1 --folds=$folds --test=$test1 --test_output=$dt_out_f --dtree/min_leaf_size=$min_leaf_size --dtree/max_leaf_size=$max_leaf_size`;

	my @dt_11;
	open IN, $dt_out_f or die "Cant open $dt_out_f...\n";

	while(<IN>) {
	    push(@dt_11, $_);
	}

	close IN;
	$to = `rm -rf $dt_out_f`;

	if ($#dt_10 != $#dt_11) {
	    die "10 = $#dt_10, 11 = $#dt_11...\n";
	}

	# computing the correct classification of class 0 $c0
	$c0 = 0;

	for ($i = 0; $i <= $#dt_00; $i++) {
	    if ($dt_00[$i] * $p0 > $dt_01[$i] * $p1) {
		$c0++;
	    }
	}

	# computing the correct classification of class 1 $c1
	$c1 = 0;

	for ($i = 0; $i <= $#dt_11; $i++) {
	    if ($dt_11[$i] * $p1 > $dt_10[$i] * $p0) {
		$c1++;
	    }
	}

	$acc = ($c0 + $c1) / ($sizes[1] + $sizes[3]);

	print "Accuracy: $acc\n$c0 / $sizes[1]\n$c1 / $sizes[3]\n";
    } else {
	$train0 = $ARGV[0];
	$train1 = $ARGV[1];
	$test = $ARGV[2];
	$folds = $ARGV[3];
	$c0_size = $ARGV[4];

	$min_leaf_size = $ARGV[5];
	$max_leaf_size = $ARGV[6];

	# computing the set sizes to compute the priors
	my @sizes;
	for ($i = 0; $i < 3; $i++) {
	    $to = `wc -l $ARGV[$i]`;
	    @a = split / /, $to;
	    $sizes[$i] = $a[0];
	    undef @a;
	}

	print "$sizes[0], $sizes[1] - $sizes[2]\n";


	$p0 = $sizes[0] / ($sizes[0] + $sizes[1]);
	$p1 = $sizes[1] / ($sizes[0] + $sizes[1]);


	$dt_out_f = "dt_out_tmp.csv";

	# computing 01on0
	$to = `./dt_driver --data=$train0 --folds=$folds --test=$test --test_output=$dt_out_f --dtree/min_leaf_size=$min_leaf_size --dtree/max_leaf_size=$max_leaf_size --dtree/use_vol_reg=false`;

	my @dt_010;
	open IN, $dt_out_f or die "Cant open $dt_out_f...\n";

	while(<IN>) {
	    push(@dt_010, $_);
	}

	close IN;
	$to = `rm -rf $dt_out_f`;

	# computing 01on1
	$to = `./dt_driver --data=$train1 --folds=$folds --test=$test --test_output=$dt_out_f --dtree/min_leaf_size=$min_leaf_size --dtree/max_leaf_size=$max_leaf_size --dtree/use_vol_reg=false`;

	my @dt_011;
	open IN, $dt_out_f or die "Cant open $dt_out_f...\n";

	while(<IN>) {
	    push(@dt_011, $_);
	}

	close IN;
	$to = `rm -rf $dt_out_f`;


	if ($#dt_010 != $#dt_011) {
	    die "010 = $#dt_010, 011 = $#dt_011...\n";
	}

	# computing the correct classification of both classes
	$c0 = 0;
	$c1 = 0;

	for ($i = 0; $i <= $#dt_010; $i++) {
	    if ($i < $c0_size) {
		if ($dt_010[$i] * $p0 > $dt_011[$i] * $p1) {
		    $c0++;
		}
	    } else {
		if ($dt_010[$i] * $p0 < $dt_011[$i] * $p1) {
		    $c1++;
		}
	    }
	}

	$acc = ($c0 + $c1) / $sizes[2];
	$c1_size = $sizes[2] - $c0_size;

	print "Accuracy: $acc\n$c0 / $c0_size\n$c1 / $c1_size\n";
    }
}
