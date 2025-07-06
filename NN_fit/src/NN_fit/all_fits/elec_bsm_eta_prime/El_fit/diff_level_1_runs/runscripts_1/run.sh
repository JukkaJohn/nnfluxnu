#!/bin/bash
	source ~/.bashrc
	conda activate nnpdf_dev
	sed -i  \
    	-e '/level_1_instance =/s/.*/level_1_instance = 1/' \
    	-e '/seed =/s/.*/seed = 1/' \
    	../../fit_diff_level1.py
	#!/bin/bash
python ../../fit_diff_level1.py
