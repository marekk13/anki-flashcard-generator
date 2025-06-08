import json
import re
import time
import datetime
from google import genai
from google.genai.errors import ClientError
from config import API_KEY

# Set your PDF file path here
FILE_PATH = r'C:\Users\Marek\Desktop\studia_magisterka\semestr 1\analiza danych w naukach o ziemi\wykłady\ADwNoZ.pdf'

PROMPT_PARSE = """
W załączniku znajduje się wykład z zajęć w postaci pliku PDF.
Twoim zadaniem jest:
- Przeczytać cały dokument, łącznie z tekstem na zdjęciach (użyj OCR).
- Na podstawie treści i tematyki oddzielić wykład na logiczne części, gdzie każda ma maksymalnie 1000 słów.
- Dla każdej sekcji wygenerować podsumowanie informacji zawartych w tej części (do 700 słów) oraz pełny tekst.
- W pierwszej linii odpowiedzi, którą wygenerujesz, umieścić informację na ile sekcji podzieliłeś wykład, w formacie "Podzielono na X sekcji".
Chodzi o łączną liczbę sekcji całego dokumentu, a nie ile wygenerujesz w tej odpowiedzi (maksymalnie 10).
- Zinterpretuj także wzory matematyczne i zapisz je w formacie MathJax (czyli np. \( a^2 + b^2 = c^2 \) ). Jeśli wzory są nieczytelne (np. na skanach), spróbuj odtworzyć ich znaczenie na podstawie kontekstu. Zadbaj o poprawną składnię MathJax, nadającą się do użycia np. w Anki.

Zwracaj sekcje w następującym formacie:

Section: <numer>
Title: <tytuł sekcji>
Slides: <numery slajdów należące do tej sekcji>
Summary:
<podsumowanie sekcji – max 700 słów>

FullText:
<pełny tekst sekcji – max 1000 słów>

Jeśli podział będzie dłuższy niż 10 sekcji, zatrzymaj się na sekcji 10 – późniejsze sekcje wygeneruję w osobnym zapytaniu.
"""

PROMPT_CONTINUE = """
Kontynuuj z następnymi 10 sekcjami (analogiczny format jak poprzednio). 
"""

PROMPT_GENERATE_FLASHCARDS = """
Chcę, abyś stworzył talię fiszek z tekstu, który udostępnię w następnym prompcie.

Instrukcje tworzenia talii fiszek:
- Fiszki powinny być proste, jasne i skoncentrowane na najważniejszych informacjach tekstu.
- Upewnij się, że pytania są konkretne i jednoznaczne.
- Używaj prostego i bezpośredniego języka, aby karty były łatwe do odczytania i zrozumienia.
- Odpowiedzi powinny zawierać tylko jeden kluczowy fakt/nazwę/pojęcie/termin.
- Fiszki mają być wygenerowane w formie CSV z kolumnami pytanie i odpowiedź (tylko te kolumny). Separatorem ma być średnik (;).
- Fiszki powinny wyczerpująco opisywać zadany tekst, jednoczesne nie powtarzając się.
- Fiszki powinny być w języku polskim.
- Jako że udostępniony tekst jest skryptem z wykładu może zawierać dodatkowe informacje, które nie zawierają wiedzy związanej z ogólnym tematem tekstów i nie są istotne do stworzenia fiszek, np. daty zaliczeń, nazwiska wykładowców, nazwy instytucji, itp. Nie uwzględniaj tych informacji w fiszkach.

Teraz stwórz fiszki na podstawie poniższego tekstu. W swojej odpowiedzi nie umieszczaj żadnych dodatkowych informacji, tylko tabelę z fiszkami.

"""

def get_genai_client():
    return genai.Client(api_key=API_KEY)

def upload_pdf(client, file_path):
    return client.files.upload(file=file_path)

def create_chat(client, model="gemini-2.5-flash-preview-05-20"):
    return client.chats.create(model=model)

def parse_slide_range(slide_str):
    slides = []
    for part in slide_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            slides.extend(range(start, end + 1))
        else:
            slides.append(int(part))
    return slides

def parse_sections(text):
    sections = []
    raw_sections = re.split(r'\nSection:\s*(\d+)', text)
    for i in range(1, len(raw_sections), 2):
        sec_num = int(raw_sections[i])
        content = raw_sections[i + 1]
        title_match = re.search(r'Title:\s*(.+)', content)
        slides_match = re.search(r'Slides:\s*(.+)', content)
        summary_match = re.search(r'Summary:\s*\n(.+?)\nFullText:', content, re.DOTALL)
        fulltext_match = re.search(r'FullText:\s*\n(.+)', content, re.DOTALL)
        sections.append({
            "id": sec_num,
            "title": title_match.group(1).strip() if title_match else "",
            "slides": parse_slide_range(slides_match.group(1)) if slides_match else [],
            "summary": summary_match.group(1).strip() if summary_match else "",
            "text": fulltext_match.group(1).strip() if fulltext_match else ""
        })
    return sections

def save_sections(sections):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    with open(f'sections_{timestamp}.json', 'w', encoding='utf-8-sig') as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)

def load_sections(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def generate_flashcards(client, text, retries=3, delay=10):
    for attempt in range(retries):
        try:
            flashcard_response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=PROMPT_GENERATE_FLASHCARDS + text,
            )
            time.sleep(delay)
            return flashcard_response.text
        except ClientError as e:
            print(f"ClientError: {e}")
            print("Wystąpił błąd podczas generowania fiszek. Spróbuję ponownie za minutę...")
            print(f"Sekcja dla której nie zostały wygenerowane fiszki:\n{text[:100]}...\n")
            time.sleep(60)
    raise RuntimeError(f"Nie udało się wygenerować fiszek po {retries} próbach.")

def save_flashcards(response, filename):
    with open(filename, 'w', encoding='utf-8-sig') as f:
        f.write(response + '\n')

def transform_to_csv(responses):
    return '\n'.join(
        line
        for response in responses
        for line in response.strip().split('\n')
        if ';' in line and 'pytanie' not in line
    )

def process_pdf_and_generate_sections(file_path):
    client = get_genai_client()
    chat = create_chat(client)
    file = upload_pdf(client, file_path)
    first_response = chat.send_message([PROMPT_PARSE, file])
    print(f"Otrzymano pierwszą odpowiedź od modelu:\n{first_response.text}\n\n")
    n_sections = first_response.text.split("\n")[0].strip().split()[-2]
    try:
        n_sections = int(n_sections)
    except ValueError:
        raise ValueError(f"Nie udało się odczytać liczby sekcji z odpowiedzi: {first_response.text}")
    finished = n_sections <= 10
    further_responses = []
    i = 2
    while not finished:
        try:
            response = chat.send_message(PROMPT_CONTINUE)
        except ClientError as e:
            time.sleep(60)
            continue
        print(f"Otrzymano {i} odpowiedź od modelu:\n{response.text}\n\n")
        further_responses.append(response.text)
        if not response or not getattr(response, "text", None):
            raise RuntimeError(f"Pusta odpowiedź od LLM. Obiekt response: {response}")
        finished = n_sections - i * 10 <= 0
        i += 1
    responses = [first_response.text] + further_responses
    sections = []
    for text in responses:
        sections.extend(parse_sections(text))
    save_sections(sections)
    return sections

def generate_and_save_flashcards(sections, prefix="flashcards"):
    client = get_genai_client()
    print("Rozpoczynam generowanie fiszek dla sekcji z prawdziwym tekstem...")
    flashcard_responses_og_text = [generate_flashcards(client, section['text']) for section in sections]
    print("Rozpoczynam generowanie fiszek dla sekcji ze streszczeniem...")
    flashcard_responses_summary = [generate_flashcards(client, section['summary']) for section in sections]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    # wyodrebij ze zmiennej flashcard_responses_og_text linie zawierające średnik (uzyj funkcji filter) i zapisz je jako pliki csv
    flashcard_responses_og_text = list(filter(lambda x: ';' in x, flashcard_responses_og_text))
    flashcard_responses_summary = list(filter(lambda x: ';' in x, flashcard_responses_summary))

    flashcard_responses_og_text = transform_to_csv(flashcard_responses_og_text)
    flashcard_responses_summary = transform_to_csv(flashcard_responses_summary)

    save_flashcards(flashcard_responses_og_text, f'{prefix}_og_text_{timestamp}.csv')
    save_flashcards(flashcard_responses_summary, f'{prefix}_summary_{timestamp}.csv')

def main():
    # Step 1: Process PDF and generate sections
    sections = process_pdf_and_generate_sections(FILE_PATH)
    # Step 2: Generate and save flashcards
    generate_and_save_flashcards(sections)

if __name__ == "__main__":
    main()