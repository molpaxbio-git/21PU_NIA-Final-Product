project="./runs/infer"
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
echo [Log] Validation started at $(date)... >> ${project}/${name}/val_log.txt
printf 'image size: '
read -r isz
printf 'batch size: '
read -r b
echo [Log] python infer.py --img ${isz} --batch ${b} --data ./nia_utils/data.yaml --device ${device} --weights ./runs/train/${1}/weights/best.pt --name ${name} --verbose --project ${project} >> ${project}/${name}/val_log.txt
python infer.py --img ${isz} --batch ${b} --data ./nia_utils/data.yaml --device ${device} --weights ./runs/train/${1}/weights/best.pt --name ${name} --verbose --project ${project}

echo [Log] Validation ended at $(date)... >> ${project}/${name}/val_log.txt