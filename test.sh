project='./runs/test'
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
echo [Log] Test started at $(date)... >> ${project}/${name}/test_log.txt
isz=224
echo [Log] python test.py --img ${isz} --device ${device} --weights ./runs/train/${name}/weights/best.pt --name ${name} --project ${project} >> ${project}/${name}/test_log.txt
python test.py --img ${isz} --device ${device} --weights ./runs/train/${name}/weights/best.pt --name ${name} --project ${project}

echo [Log] Test ended at $(date)... >> ${project}/${name}/test_log.txt