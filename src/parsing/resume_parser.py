import io
import PyPDF2
import docx


def extract_text_from_pdf(file_bytes):
    """Extract text from a PDF file's bytes."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_bytes):
    """Extract text from a DOCX file's bytes."""
    doc = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join(para.text for para in doc.paragraphs)
    return text


def extract_resume_text(uploaded_file):
    """
    Takes a Streamlit UploadedFile object and returns plain text,
    dispatching based on file extension.
    """
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {filename}")