#!/bin/bash

if [ $# -ne 1 ]; then
	echo "usage: $0 <#extraciton_number>"
	exit 1;
else
	echo "Beginning extraction..."
fi

DATA_EXTRACTION_NO=$1
EXTRACTION_FOLDER="dataset_$DATA_EXTRACTION_NO"
mkdir $EXTRACTION_FOLDER

for dir in `ls dataset`; do
	DIR="dataset/$dir"
	for file in `ls $DIR`; do
		cp $DIR/$file $EXTRACTION_FOLDER/${dir}_${file}
	done;
done;

for file in $EXTRACTION_FOLDER/*.csv; do
	sed -i "s/\//\_/g" $file
done;

BBPATH=$EXTRACTION_FOLDER/bboxes.csv
touch $BBPATH

for file in $EXTRACTION_FOLDER/*.csv; do
	tail -n +2 $file >> $BBPATH
done;

sed -i "s/\(.*\)\([A-Z]\)\(.*\)/\1\2\3,\2/g" $EXTRACTION_FOLDER/bboxes.csv

cp class_to_id.csv $EXTRACTION_FOLDER
echo "Extraction ended..."
echo "Image stats"
file $EXTRACTION_FOLDER/*.jpg | cut -d "," -f8 | sort | uniq -c
echo "Classes stats: "
cut -d "," -f6 $EXTRACTION_FOLDER/bboxes.csv | sort | uniq -c | column -n

