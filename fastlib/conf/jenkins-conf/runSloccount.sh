#!/bin/bash

DATA_FILE="sloccount.sc"
DATA_DIR=".slocdata"

if [ ! -d $DATA_DIR ]
then
  mkdir $DATA_DIR
fi

sloccount --wide --details --datadir .slocdata repo > $DATA_FILE

mv $DATA_FILE $DATA_DIR
