#! /bin/bash
while true; do
  for file in `ls models/Ant-v2` #注意此处这是两个反引号，表示运行系统命令
  do
    #echo $file #在此处处理文件即可
    nohup python -u Test.py --file_name $file > test_dir/$file.log 2>&1 &
    sleep 10s
    mv models/Ant-v2/${file} achieve_dir
  done
done

  

