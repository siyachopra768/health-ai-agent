from pypdf import PdfReader
import re

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_lab_values(text):
    data = {}
    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        match = re.search(r"([A-Za-z ()]+)\s+([\d.]+)\s+([a-zA-Z/%^0-9µ]+)\s+([\d.]+[-–][\d.]+)", line)

        if match:
            name, value, unit, ref = match.groups()

            try:
                low, high = re.split(r"[-–]", ref)

                data[name.strip()] = {
                    "value": float(value),
                    "unit": unit,
                    "ref_low": float(low),
                    "ref_high": float(high)
                }
            except:
                continue

    return data