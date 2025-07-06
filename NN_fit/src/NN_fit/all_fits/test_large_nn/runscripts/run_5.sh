#!/bin/bash
source ~/.bashrc
conda activate nnpdf_dev
sed -i '/level_1_instance =/s/.*/level_1_instance = 5/' ../fit_faser_data.py
#!/bin/bash
python ../fit_faser_data.py
