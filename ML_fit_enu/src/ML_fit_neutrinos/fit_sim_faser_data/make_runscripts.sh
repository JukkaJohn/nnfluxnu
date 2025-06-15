#!/bin/bash
REPS=$1

rm pred.txt
rm chi_square.txt
rm mub_pdf.txt
rm mu_pdf.txt
rm events.txt
rm chi_squares_for_postfit.txt
rm -r runscripts
rm fit_report_sim.txt
rm train_indices.txt
rm val_indices.txt
rm training_lengths.txt
rm scripts_faser_sim.txt
mkdir runscripts  

cd runscripts

cp ../jobsubmit.sub .
python_path=$(which python)
RUN_CONTENT="#!/bin/bash\npython ../fit_sim_data.py"

#for i in $(seq 1 $REPS)
#do	
echo -e "#!/bin/bash\nsource ~/.bashrc\nconda activate nnpdf_dev\nsed -i '/level_1_instance =/s/.*/level_1_instance = 42/' ../fit_sim_data.py\n$RUN_CONTENT" > "run.sh"
chmod +x "run.sh"	
#done


pwd
rm scripts_faser_sim.txt
for i in $(seq 1 $REPS)
do
        echo "./run.sh" >> scripts_faser_sim.txt
done

echo "$REPS run.sh files created successfully!"










