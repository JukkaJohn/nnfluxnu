find . -type f \( -name "fit_epos.py" -o -name "postfit_analysis.py" \) -exec sed -i '' 's/pdf_name = "FASERv_Run3_EPOS+POWHEG_7TeV"/pdf_name = "FASER_2412.03186_EPOS+POWHEG_7TeV"/g' {} +

