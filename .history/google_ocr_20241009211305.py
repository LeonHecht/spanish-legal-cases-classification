from utils import ocr
import pandas as pd
from tqdm import tqdm

path = 'corpus/descargados/'

# get all the PDF files in the directory
import os
pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]

df_google_ocr = pd.DataFrame(columns=['codigo', 'text'])

for file in tqdm(pdf_files[400:], desc="Processing PDFs"):
    
    text = ocr.pdf2text(path + file)
    codigo = file.split('.')[0]

    # Create a new row as a DataFrame
    new_row = pd.DataFrame({'codigo': [codigo], 'text': [text]})
    
    # Concatenate the new row with the existing dataframe
    df_google_ocr = pd.concat([df_google_ocr, new_row], ignore_index=True)

# Save the dataframe to a CSV file for future use
df_google_ocr.to_csv('google_ocr_results.csv', index=False)

print("All files processed and saved to google_ocr_results.csv.")
