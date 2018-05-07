#!/bin/bash

rm -rf super_dataset && \
	mkdir super_dataset && \
	mkdir super_dataset/train && \
	mkdir super_dataset/validation && \
	touch super_dataset/train/bboxes.csv && \
	touch super_dataset/validation/bboxes.csv

