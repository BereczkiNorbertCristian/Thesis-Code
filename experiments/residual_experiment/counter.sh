
sum=0
for x in {A..D}; do
	res=`bash count.sh $x`
	sum=$((sum + res))
done
echo "sum is $sum" 


