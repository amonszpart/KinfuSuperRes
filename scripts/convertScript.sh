#!/bin/bash
for f in img8*.png; do 
	echo "Processing $f file.."; 
	NAME=`echo "$f" | cut -d'.' -f1`; 
	echo $NAME;
	convert $f $NAME.pgm 
done

