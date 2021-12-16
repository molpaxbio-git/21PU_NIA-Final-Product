cat /proc/cpuinfo | grep "model name" | head -1 | awk '{{printf "CPU: "};for(i=4;i<=NF;i++){printf "%s ", $i};{print ""};}'
nvidia-smi --query | fgrep 'Product Name' | awk '{{printf "GPU: "};for(i=4;i<=NF;i++){printf "%s ", $i};{print ""};}'
free -h | head -2 | tail -1 | awk '{printf "RAM: "};{print $2}'
df -h -T | grep dev/sdb | awk '{printf "HDD: "}{print $3}'
lsb_release -d | awk '{{printf "OS: "};for(i=2;i<=NF;i++){printf "%s ", $i};{print ""};}'
git -C $(pwd) describe --tags --long --always | awk '{printf "YOLOv5 git version: "}{print $0}'
echo Python frameworks version
cat requirements.txt