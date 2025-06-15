#!/bin/bash
	source ~/.bashrc
	conda activate nnpdf_dev
	sed -i  \
    	-e '/level_1_instance =/s/.*/level_1_instance = 2/' \
    	-e '/seed =/s/.*/seed = 2/' \
    	../../fit_diff_level1.py
	#!/bin/bash
python ../../fit_diff_level1.py
