#!/bin/bash
REPS=$1

rm -r runscripts
mkdir runscripts  

cd runscripts

cp ../jobsubmit.sub .
python_path=$(which python)
RUN_CONTENT="#!/bin/bash\npython ../fit_sibyll.py"

#for i in $(seq 1 $REPS)
#do		
echo -e "#!/bin/bash\nsource ~/.bashrc\nconda activate nnpdf_dev\nsed -i '/level_1_instance =/s/.*/level_1_instance = 42/' ../fit_sibyll.py\n$RUN_CONTENT" > "run.sh"
chmod +x "run.sh"
#done


pwd
rm scripts.txt
for i in $(seq 1 $REPS)
do
        echo "./run.sh" >> scripts.txt
done

echo "$REPS run.sh files created successfully!"










