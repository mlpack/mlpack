#!/usr/bin/perl -w
# This perl script removes duplicate libraries from the right to the left and
# removes duplicate -L library paths from the left to the right
use strict;

my @all_libs = @ARGV;
#
# Move from left to right and remove duplicate -l libraries
#
my @cleaned_up_libs_first;
foreach( reverse @all_libs ) {
	$_ = remove_rel_paths($_);
	if( $_=~/-L/ ) {
		unshift @cleaned_up_libs_first, $_;
	}
	else {
		if( !entry_exists($_,\@cleaned_up_libs_first) ) {
			unshift @cleaned_up_libs_first, $_;
		}
	}
}

#
# Move from right to left and remove duplicate -L library paths
#
my @cleaned_up_libs;
foreach( @cleaned_up_libs_first ) {
	$_ = remove_rel_paths($_);
	if( !($_=~/-L/) ) {
		push @cleaned_up_libs, $_;
	}
	elsif( !entry_exists($_,\@cleaned_up_libs) ) {
		push @cleaned_up_libs, $_;
	}
}
#
# Print the new list of libraries and paths
#
print join( " ", @cleaned_up_libs );

#
# Subroutines
#
sub entry_exists {
	my $entry = shift; # String
	my $list  = shift; # Reference to an array
	foreach( @$list ) {
		if( $entry eq $_ ) { return 1; }
	}
	return 0;
}
#
sub remove_rel_paths {
	my $entry_in = shift;
	if ($entry_in=~/-L\.\./) {
		return $entry_in;
	}
	my @paths = split("/",$entry_in);
	my @new_paths;
	foreach( @paths ) {
		if( !($_=~/\.\./) ) {
			push @new_paths, $_;
		}
		else {
			pop @new_paths
		}
	}
	return join("/",@new_paths);
}
