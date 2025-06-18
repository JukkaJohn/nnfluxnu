#!/bin/bash
NUM_DIRS=$1
NUM_RUNS=$2



python_path=$(which python)
RUN_CONTENT="#!/bin/bash\npython ../../fit_sim_data.py"
cd ..
rm fit_report_diff_level_1.txt
cd diff_level_1_runs

rm -r runscripts_*/
for dir_index in $(seq 1 $NUM_DIRS); do
        dir_name="runscripts_$dir_index"
	mkdir "$dir_name"
        cd "$dir_name"
	
	echo -e "#!/bin/bash
	source ~/.bashrc
	conda activate nnpdf_dev
	sed -i  \\
    	-e '/level_1_instance =/s/.*/level_1_instance = $dir_index/' \\
    	-e '/seed =/s/.*/seed = $dir_index/' \\
    	../../fit_sim_data.py
	$RUN_CONTENT" > "run.sh"

        # echo -e "#!/bin/bash\nsource ~/.bashrc\nconda activate nnpdf_dev\n$RUN_CONTENT" > "run.sh"
        chmod +x "run.sh"

	> scripts_level_1_faser_sim.txt
	for _ in $(seq 1 $NUM_RUNS); do
		echo "./run.sh" >> scripts_level_1_faser_sim.txt
	done
	cp ../jobsubmit.sub .
	cd ..

done

echo "$NUM_DIRS directories with $NUM_RUNS entries in scripts_faser_sim.txt each created successfully!"
