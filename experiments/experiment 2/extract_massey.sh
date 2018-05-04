#!/bin/bash
MASSEY_FOLDER='dataset_massey'

if [ ! -e $MASSEY_FOLDER/bboxes.csv ]; then
	touch $MASSEY_FOLDER/bboxes.csv
fi

cd $MASSEY_FOLDER && file *.png | sed "s/:/,/g" | cut -d "," -f1,3 > bboxes.csv && cd ..
python render_massey_bb.py $MASSEY_FOLDER/bboxes.csv
