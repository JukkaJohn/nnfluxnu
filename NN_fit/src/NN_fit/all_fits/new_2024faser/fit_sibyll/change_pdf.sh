find . -type f \( -name "fit_sibyll.py" -o -name "postfit_analysis.py" \) -exec sed -i '' 's/pdf_name = "FASERv_Run3_SIBYLL+SIBYLL_7TeV"/pdf_name = "FASER_2412.03186_SIBYLL+SIBYLL_7TeV"/g' {} +

