# Lecture PDF Sectioning and Flashcard Generator

This project processes lecture PDFs, splits them into logical sections, summarizes each section, and generates flashcards in CSV format using Google Gemini API.

## Features

- Reads and processes PDF lecture files (including OCR for images).
- Splits the document into logical sections (max 1000 words each).
- Summarizes each section (up to 700 words).
- Extracts and formats mathematical formulas in MathJax.
- Generates flashcards (Q&A) in CSV format for each section and summary.

## Requirements

- Python 3.8+
- Google AI Studio API access

## Setup

1. **Clone the repository** 

2. **Install required libraries**:
    ```bash
    pip install google-generativeai
    ```
3. **Ensure you have access to the Google Gemini API** and have your API key ready. Use links below to get started:
   - https://aistudio.google.com/welcome
   - https://aistudio.google.com/apikey

4. **Create a `config.py` file** in the project root with your Google Gemini API key:
    ```python
    API_KEY = 'your_google_gemini_api_key'
    ```

5. **Set the path to your PDF file** in `main.py`:
    ```python
    FILE_PATH = r'path\to\your\lecture.pdf'
    ```

6. **Run the main script:**
    ```bash
    python main.py
    ```

## Output

- Sectioned lecture content is saved as a timestamped JSON file.
- Flashcards are saved as CSV files (one for full text read by LLM, one for summaries made by the model).

## Notes

- The script will process up to 10 sections at a time; if your document is longer, it will continue in batches, so processing time may be longer.
- Flashcards are generated in Polish, as per the prompt instructions.

## License

This project is for educational purposes.