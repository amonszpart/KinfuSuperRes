#!/bin/bash
if [[ $# -lt 3 ]]
then
    echo "Usage: bruteYang.sh 'directory' 'depth file prefix' 'file extension'";
    exit 0;
fi

dirName=$1;
for f in `ls $dirName | grep -e "^$2" | grep -e "png"`
do
  depName=$f;
  imgName=${f:1};

  echo "Processing $depName and $imgName files...";
  fun="/home/bontius/KinfuSuperRes/Tsdf_vis/build/tsdf_vis --yangd $dirName --dep $depName --img $imgName --brute-force";
  eval $fun
done

# ./bruteYang.sh ~/workspace/rec/testing/ram_20130818_1209_lf_200/poses "d*.png"
# ~/KinfuSuperRes/Tsdf_vis/bruteYang.sh '..' "d" "png"

