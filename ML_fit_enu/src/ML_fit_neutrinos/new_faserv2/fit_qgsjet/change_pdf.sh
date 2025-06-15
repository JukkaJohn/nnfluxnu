find . -type f \( -name "fit_qgsjet.py" -o -name "postfit_analysis.py" \) -exec sed -i '' 's/pdf_name = "FASERv_Run3_QGSJET+POWHEG_7TeV"/pdf_name = "FASERv2_QGSJET+POWHEG_7TeV"/g' {} +

