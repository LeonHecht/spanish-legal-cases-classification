from google.cloud import vision
import io
from google.oauth2 import service_account
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import re
import logging

# Suppress logging messages from PyPDF2
logging.getLogger("PyPDF2").setLevel(logging.ERROR)


def get_text_from_image(image_path) -> str:
    """
    Extract text from an image using Google Cloud Vision API

    Parameters
    ----------
    image_path : str
        Path to the image file

    Returns
    -------
    str
        Extracted text from image    
    """    
    # Explicitly load the credentials
    credentials = service_account.Credentials.from_service_account_file('/home/leon/tesis/Clasificacion_Sentencias/GoogleCloudKey/strong-skyline-423405-a8-52bcbd50a7e1.json')

    # Set up Google Cloud Vision client
    client = vision.ImageAnnotatorClient(credentials=credentials)

    # Load the image file
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform OCR
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(f'{response.error.message}')
    
    text = response.full_text_annotation.text
    return text


def get_text_from_scanned_pdf(pdf_path) -> str:
    """
    Extract text from a PDF file using Google Cloud Vision API

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file
    
    Returns
    -------
    str
        Extracted text from PDF
    """
    # Convert PDF to a list of images, one per page
    pages = convert_from_path(pdf_path)

    # Initialize an empty string to hold all the text
    full_text = ""

    # Iterate through all pages and process each image with OCR
    for page_number, page_image in enumerate(pages, start=1):
        print(f"Processing page {page_number}...")
        
        # Save the page image temporarily
        temp_image_path = f"temp_page_{page_number}.jpg"
        page_image.save(temp_image_path, "JPEG")
        
        # Extract text from the image
        text = get_text_from_image(temp_image_path)
        
        # Append the text to the full text
        full_text += text + "\n"

    return full_text


def pdf2text(pdf_path) -> str:
    """
    Extract text from a PDF file

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file
    
    Returns
    -------
    str
        Extracted text from PDF
    """
    # Cargar el PDF
    reader = PdfReader(pdf_path)
    text_total = ""

    # Iterar sobre las primeras páginas para buscar texto
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():  # Si se encuentra texto
            text_total += text
    sentence = "Para conocer la validez del documento, verifique aquí"
    sentence_regex = re.sub(r'\s+', r'\\s*', sentence)
    if re.search(sentence_regex, text_total):
        return text_total
    else:
        return get_text_from_scanned_pdf(pdf_path)
    

def count_pages(pdf_path) -> int:
    """
    Count the number of pages in a PDF file

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file
    
    Returns
    -------
    int
        Number of pages in the PDF
    """
    reader = PdfReader(pdf_path)
    return len(reader.pages)


def is_scanned_pdf(pdf_path) -> bool:
    """
    Check if a PDF file is a scanned document

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file
    
    Returns
    -------
    bool
        True if the PDF is scanned, False otherwise
    """
    # Cargar el PDF
    reader = PdfReader(pdf_path)
    text_total = ""

    # Iterar sobre las primeras páginas para buscar texto
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():  # Si se encuentra texto
            text_total += text
    sentence = "Para conocer la validez del documento, verifique aquí"
    sentence_regex = re.sub(r'\s+', r'\\s*', sentence)
    if re.search(sentence_regex, text_total):
        return False
    else:
        return True
    