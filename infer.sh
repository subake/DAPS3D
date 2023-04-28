#!/bin/sh
helpFunction() {
   echo "Options not found"
   exit 1
}

get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

while getopts "d:f:p:m:s:l:c:" opt
do
   case "$opt" in
      d ) d="$OPTARG" ;;
      f ) f="$OPTARG" ;;
      p ) p="$OPTARG" ;;
      m ) m="$OPTARG" ;;
      s ) s="$OPTARG" ;;
      l ) l="$OPTARG" ;;
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
l=$(get_abs_filename "$l")
p=$(get_abs_filename "$p")

export CUDA_VISIBLE_DEVICES="$c"

cd ./train/tasks/semantic/

./infer.py -d "$d" \
           -f "$f" \
           -l "$l" \
           -p "$p" \
           -m "$m" \
           -s "$s"

# ./infer.sh -d /Dataset/ -f cfgs/ -l ./logs/ddrnet_aug5_T/ -m ddrnet -p ./logs/infer/ddrnet_aug5_T -s valid -c 0
