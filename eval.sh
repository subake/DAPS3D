#!/bin/sh

helpFunction() {
   echo "Options not found"
   exit 1
}

get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

while getopts "d:f:p:s:" opt
do
   case "$opt" in
      d ) d="$OPTARG" ;;
      f ) f="$OPTARG" ;;
      p ) p="$OPTARG" ;;
      s ) s="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done

if [ -z "$d" ] || [ -z "$p" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

f=$(get_abs_filename "$f")
d=$(get_abs_filename "$d")
p=$(get_abs_filename "$p")

cd ./train/tasks/semantic/

./evaluate_iou.py -d "$d" \
                  -f "$f" \
                  -p "$p" \
                  -s "$s"

# ./eval.sh -d /Dataset/ -f cfgs/ -p ./logs/infer/ddrnet_aug5_T -s valid
