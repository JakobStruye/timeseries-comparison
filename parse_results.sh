grep reset | sed 's/}/ /g' | sed 's/,/ /g'| awk '{ print $14 " " $6 }'
