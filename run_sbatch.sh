#!/bin/bash
#cp ../00_repo/** ../00_ae/ -r
#cp ../00_repo/** ../00_ae_repo/ -r
git log -1 --format="%H" > /home/ngw861/00_ae/git_commit.txt
git --git-dir=/home/ngw861/00_ae_repo/.git --work-tree=/home/ngw861/00_ae_repo pull
echo "Updated Git"

if [[ -f "$1" || -d "$1" ]]
then
    echo "The First argument is a file or directory"
fi

while getopts ":d:s:a:" opt; do
  case $opt in
    d) dep="$OPTARG"
      echo "-d (dependency) was triggered, Parameter: $OPTARG" >&2
      ;;
    a) adep="$OPTARG"
      echo "-a (dependency 5 mins after x starts) was triggered, Parameter: $OPTARG" >&2
      ;;
    s) suf="$OPTARG"
      echo "-s (suffix) was triggered, Parameter: $OPTARG" >&2
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
ARGS=""
if [ -z "$dep" ]
then 
    echo "dependency -d argument is empty"
else
    ARGS+="--dependency=afterany:$dep "
fi
if [ -z "$adep" ]
then 
    echo "dependency -a argument is empty"
else
    ARGS+="--dependency=after:$adep+5 "
fi
if [ -z "$suf" ]
then 
    echo "suffix -s argument is empty"
else
    ARGS+="--export=EXP_SUFFIX=$suf"
fi

echo "running with args: $ARGS"

sbatch $ARGS sbatch.sh
