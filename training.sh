project="./runs/train"
name=$1
device=$2
if [ -d ${project}/${name} ] ; then
    echo "Directory ${project}/${name} exists."
    while :
    do
        printf 'Overwrite it? [Y/N] '
        read -r answer
        if [[ ${answer} =~ ^(Y|y|N|n)$ ]]; then
            if [[ ${answer} =~ ^(Y|y)$ ]]; then
                rm -r ${project}/${name}
                echo "Project ${project}/${name} has been overwritten."
            else
                num=2
                while [[ -d ${project}/${name}_${num} ]];
                do
                    num=$(expr ${num} + 1) 
                done
                name="${name}_${num}"
                echo "New project ${project}/${name} has been created."
            fi
            mkdir ${project}/${name}
            break
        else
            echo "Please press Y(y) or N(n) and enter"
        fi
    done
else
    echo "Directory ${project}/${name} not exists. Create new one."
    mkdir ${project}/${name}
fi
echo [Log] Training started at $(date)... >> ${project}/${name}/training_log.txt
printf 'image size: '
read -r isz
printf 'batch size: '
read -r b
printf 'max epochs: '
read -r ep
echo [Log] python training.py --img ${isz} --batch ${b} --epochs ${ep} --device $2 --cfg ./models/sP5.yaml --weights yolov5s.pt --name ${name} --project ${project} >> ${project}/${name}/training_log.txt
python training.py --img ${isz} --batch ${b} --epochs ${ep} --device $2 --cfg ./models/sP5.yaml --weights yolov5s.pt --name ${name} --project ${project}

echo [Log] Training ended at $(date)... >> ${project}/${name}/training_log.txt