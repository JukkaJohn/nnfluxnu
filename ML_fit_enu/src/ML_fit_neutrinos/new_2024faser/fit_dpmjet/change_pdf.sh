find . -type f \( -name "fit_dpmjet.py" -o -name "postfit_analysis.py" \) -exec sed -i '' 's/pdf_name = "FASERv_Run3_DPMJET+DPMJET_7TeV"/pdf_name = "FASER_2412.03186_DPMJET+DPMJET_7TeV"/g' {} +

