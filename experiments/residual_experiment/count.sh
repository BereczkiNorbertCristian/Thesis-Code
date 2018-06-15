
sum=0
for x in {a..z}; do
	res=`ls raw_dataset/$1/$x 2> /dev/null | wc -l`
	sum=$((sum + res))
done 
echo "$sum"
