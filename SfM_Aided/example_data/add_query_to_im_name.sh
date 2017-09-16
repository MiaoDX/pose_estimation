#!/bin/sh  
for files in $(ls *.jpg)  
    do mv $files "query_"$files  
done