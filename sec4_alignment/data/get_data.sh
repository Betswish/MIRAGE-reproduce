#!/bin/bash

for (( i=0; i<200; i=i+1 ));
do
	echo $i >> out_concat.txt
	python check_same_prediction.py >> log_concat.txt
	cd xor_attriqa
	python shuffle_reform_concat.py 
	cd ..
done
