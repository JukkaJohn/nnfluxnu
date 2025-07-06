#!/bin/bash
REPS=$1


rm chi_square.txt
rm mub_pdf.txt
rm mu_pdf.txt
rm events.txt
rm chi_squares_for_postfit.txt
rm pred.txt
rm fit_report.txt
rm training_lengths.txt
rm val_indices.txt
rm train_indices.txt
rm -r runscripts
mkdir runscripts

cd runscripts

cp ../jobsubmit.sub .
python_path=$(which python)
RUN_CONTENT="#!/bin/bash\npython ../fit_faser_data.py"

for i in $(seq 1 $REPS)
do
	echo -e "#!/bin/bash\nsource ~/.bashrc\nconda activate nnpdf_dev\nsed -i '/level_1_instance =/s/.*/level_1_instance = $i/' ../fit_faser_data.py\n$RUN_CONTENT" > "run_$i.sh"
	chmod +x "run_$i.sh"
done


pwd
rm scripts_faser_fit.txt
for i in $(seq 1 $REPS)
do
        echo "./run_$i.sh" >> scripts_faser_fit.txt
done

echo "$REPS run.sh files created successfully!"
