#!/bin/bash
source ~/.bashrc
conda activate nnpdf_dev
<<<<<<< HEAD
sed -i '/level_1_instance =/s/.*/level_1_instance = 42/' ../fit_sibyll.py
=======
>>>>>>> bf5bffb69ec4a318117f0a4aa6bdaab926d87e87
#!/bin/bash
python ../fit_sibyll.py
