#!/bin/sh
helpFunction() {
   echo "Options not found"
   exit 1
}

get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

while getopts "d:f:p:m:s:n:c:" opt
do
   case "$opt" in
      d ) d="$OPTARG" ;;
      f ) f="$OPTARG" ;;
      p ) p="$OPTARG" ;;
      m ) m="$OPTARG" ;;
      s ) s="$OPTARG" ;;
      n ) n="$OPTARG" ;;
      c ) c="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done

if [ -z "$d" ] || [ -z "$p" ] || [ -z "$m" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

f=$(get_abs_filename "$f")
d=$(get_abs_filename "$d")
p=$(get_abs_filename "$p")
m=$(get_abs_filename "$m")

export CUDA_VISIBLE_DEVICES="$c"

cd ./train/tasks/semantic/

./infer.py -d "$d" \
           -f "$f" \
           -l "$p" \
           -m "$m" \
           -n "$n" \
           -s "$s"

echo "finishing infering.\n Starting evaluating"

# ./evaluate_iou.py -d "$d" \
#                   -p "$p" \
#                   -s "$s" \
#                   -m "$m"

# ./eval.sh -d /Datasets/multisalsa/ -f cfgs/ -m ./logs/ddrnet_aug5_T/ -n ddrnet -s valid -l ./logs/infer/ddrnet_aug5_T
