#conda activate gail 
for ((i=0; i<4; i ++))
do
	nohup python -u main.py --seed $i > log_dir/seed_${i}.log 2>&1 &
done