#!/bin/bash
source ~/.bashrc
conda activate nnpdf_dev
sed -i '/level_1_instance =/s/.*/level_1_instance = 42/' ../fit_epos.py
#!/bin/bash
python ../fit_epos.py
