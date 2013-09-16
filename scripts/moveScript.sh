#!/bin/bash
for f in $1/*.png; do
        echo "Processing $f file..";
        NUMBER=`echo "$f" | cut -d'_' -f2`;
        F=`echo "$2/img8_$NUMBER"`;
	cp $F $1/
done

