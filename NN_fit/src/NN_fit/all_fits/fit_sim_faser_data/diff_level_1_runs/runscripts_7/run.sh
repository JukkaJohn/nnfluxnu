#!/bin/bash
	source ~/.bashrc
	conda activate nnpdf_dev
	sed -i  \
    	-e '/level_1_instance =/s/.*/level_1_instance = 7/' \
    	-e '/seed =/s/.*/seed = 7/' \
    	../../fit_sim_data.py
	#!/bin/bash
python ../../fit_sim_data.py
