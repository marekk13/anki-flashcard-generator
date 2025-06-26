import json
import os
import re
import time
import datetime

from google import genai
from google.genai.errors import ClientError, ServerError
from config import API_KEY

# Set your PDF file path here
# FILE_PATH = r'C:\Users\Marek\Desktop\studia_magisterka\semestr 1\analiza danych w naukach o ziemi\wykłady\W_merged.pdf'
FILE_PATH = r'C:\Users\Marek\Desktop\studia_magisterka\semestr 1\geostatystyka\wykłady\GiAD_merged_p2.pdf'

PROMPT_PARSE = """
W załączniku znajduje się wykład z zajęć w postaci pliku PDF.
Twoim zadaniem jest:
- Przeczytać cały dokument, łącznie z tekstem na zdjęciach (użyj OCR).
- Na podstawie treści i tematyki oddzielić wykład na logiczne części, gdzie każda ma maksymalnie 1200 słów.
- Dla każdego slajdu zawierającego obraz (mapę, wykres, diagram), krótko opisz, co ten obraz przedstawia, np. "Slajd 64 pokazuje diagram semiwariogramu z zaznaczonymi komponentami: nugget, sill, range".
- Dla każdej sekcji wygenerować podsumowanie informacji zawartych w tej części (do 800 słów) oraz pełny tekst.
- W pierwszej linii odpowiedzi, którą wygenerujesz, umieścić informację na ile sekcji podzieliłeś wykład, w formacie "Podzielono na X sekcji".
Chodzi o łączną liczbę sekcji całego dokumentu, a nie ile wygenerujesz w tej odpowiedzi (maksymalnie 10).
- Zinterpretuj także wzory matematyczne i zapisz je w formacie MathJax (czyli np. \( a^2 + b^2 = c^2 \) ). Jeśli wzory są nieczytelne (np. na skanach), spróbuj odtworzyć ich znaczenie na podstawie kontekstu. Zadbaj o poprawną składnię MathJax, nadającą się do użycia np. w Anki.

Zwracaj sekcje w następującym formacie:

Section: <numer>
Title: <tytuł sekcji>
Slides: <numery slajdów należące do tej sekcji>
Summary:
<podsumowanie sekcji – max 800 słów>

FullText:
<pełny tekst sekcji – max 1200 słów>

Jeśli podział będzie dłuższy niż 10 sekcji, zatrzymaj się na sekcji 10 – późniejsze sekcje wygeneruję w osobnym zapytaniu.
"""

PROMPT_CONTINUE = """
Kontynuuj z następnymi 10 sekcjami (analogiczny format jak poprzednio). 
"""

PROMPT_GENERATE_FLASHCARDS = """
Chcę, abyś stworzył talię fiszek z tekstu, który udostępnię w następnym prompcie.

Instrukcje tworzenia talii fiszek:
- Fiszki powinny być proste, jasne i skoncentrowane na najważniejszych informacjach tekstu.
- Pytanie powinno być jak najkrótsze i wymagać aktywnego przypomnienia sobie odpowiedzi. Odpowiedź powinna zawierać wyjaśnienie.
  - ŹLE: "Jaka metoda poprawia jakość estymatorów z drzewa, jest metodą samowsporną i nazywana jest 'zbiorową mądrością'?" ; "Bagging"
- Używaj prostego i bezpośredniego języka.
- Fiszki mają być wygenerowane w formie CSV z kolumnami pytanie i odpowiedź (tylko te kolumny). Separatorem ma być średnik (;).
- Fiszki powinny wyczerpująco opisywać zadany tekst, jednocześnie nie powtarzając się.
- Fiszki powinny być w języku polskim.
- Jeśli w tekście pojawiają się wzory, umieść je w odpowiedziach w formacie MathJax (np. \( a^2 + b^2 = c^2 \) ).
- Ignoruj informacje nieistotne merytorycznie (np. dane organizacyjne, nazwiska, które nie są związane z kluczową teorią).
- Jeśli pytanie dotyczy elementu wizualnego (mapy, wykresu), zaznacz to w pytaniu i podaj numer slajdu, np. "[PYTANIE WIZUALNE, slajd 57]".

Teraz stwórz fiszki na podstawie poniższego tekstu. W swojej odpowiedzi nie umieszczaj żadnych dodatkowych informacji, tylko tabelę z fiszkami.
"""

PROMPT_GENERATE_ADVANCED_FLASHCARDS = """
Twoim zadaniem jest stworzenie zaawansowanej talii fiszek z udostępnionego tekstu, który jest podsumowaniem całego wykładu. Fiszki te mają testować głębsze zrozumienie, zdolność do syntezy i porównywania koncepcji.

**Instrukcje Ogólne:**
- Pytanie powinno być zwięzłe i prowokować do myślenia. Odpowiedź powinna być wyczerpująca i dobrze wyjaśniać zagadnienie.
- Unikaj pytań, na które odpowiedź to jedno słowo.
- Używaj formatu CSV z separatorem w postaci średnika (;).
- Używaj formatu MathJax dla wzorów.
- Jeśli pytanie dotyczy elementu wizualnego (mapy, wykresu), zaznacz to w pytaniu i podaj numer slajdu, np. "[PYTANIE WIZUALNE, slajd 57]".

**Kategorie Pytań do Wygenerowania:**
Wygeneruj fiszki z następujących kategorii, aby zapewnić różnorodność i głębię nauki:

1.  **Pytania Porównawcze/Syntetyzujące:**
    -   Mają na celu porównanie dwóch lub więcej koncepcji, metod lub narzędzi.
    -   Powinny pytać o podobieństwa, różnice, wady i zalety.
    -   Przykład: "Porównaj mechanizm działania Random Forest i Gradient Boosting, wskazując na kluczowe różnice w budowie drzew i celu uczenia."

2.  **Pytania Koncepcyjne/Wyjaśniające ("Dlaczego?"):**
    -   Mają na celu sprawdzenie zrozumienia przyczyn i mechanizmów.
    -   Powinny zaczynać się od "Wyjaśnij, dlaczego...", "W jaki sposób...", "Jaki problem rozwiązuje...".
    -   Przykład: "Wyjaśnij, dlaczego modele oparte na drzewach decyzyjnych nie potrafią ekstrapolować i jakie ma to praktyczne konsekwencje w analizie danych przestrzennych."

3.  **Pytania Praktyczne/Zastosowanie (Oparte na przykładach z wykładu):**
    -   Mają na celu połączenie teorii z konkretnymi przykładami omówionymi w materiale.
    -   Powinny odnosić się do analizy smogu, danych kolarskich, funkcji sinc itp.
    -   Przykład: "Opisz, jak w przykładzie z danymi kolarskimi wykorzystano regresję logistyczną i macierz błędów do zbudowania i oceny klasyfikatora wysiłku tlenowego/beztlenowego."

Przykład:
Pytanie;Odpowiedź
Porównaj kryteria informacyjne AIC i BIC.;Oba kryteria równoważą dopasowanie modelu (\( \ln(L) \)) i jego złożoność (liczbę parametrów \( k \)). Różnią się karą: w AIC jest stała (2), a w BIC zależy od liczby danych (\( \ln(n) \)). Sprawia to, że BIC jest bardziej "krytyczny" i preferuje prostsze modele, podczas gdy AIC jest lepszy w wyborze modelu o najlepszych zdolnościach predykcyjnych.
Wyjaśnij, dlaczego w algebrze map konieczne jest skalowanie wartości przed użyciem niektórych funkcji odległości?;Ponieważ funkcje odległości, takie jak dystans Euklidesowy, są wrażliwe na skalę zmiennych. Jeśli jedna cecha (np. współrzędna geograficzna) ma znacznie większy zakres wartości niż inna, zdominuje ona obliczenia odległości, prowadząc do błędnych wyników. Skalowanie (np. normalizacja) sprowadza wszystkie cechy do wspólnego zakresu, zapewniając im równy wkład.

Teraz stwórz zaawansowane fiszki na podstawie poniższego tekstu. W swojej odpowiedzi nie umieszczaj żadnych dodatkowych informacji, tylko tabelę z fiszkami.
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
    with open(f'output/sections_{timestamp}.json', 'w', encoding='utf-8-sig') as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)

def load_sections(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def generate_flashcards(client, text, prompt, retries=3, delay=10):
    for attempt in range(retries):
        try:
            flashcard_response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt + text,
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
    if not os.path.exists('output'):
        os.makedirs('output')
    filename = os.path.join('output', filename)
    with open(filename, 'w', encoding='utf-8-sig') as f:
        f.write(response + '\n')

def transform_to_csv(responses):
    return '\n'.join(
        line
        for response in responses
        for line in response.strip().split('\n')
        if ';' in line and 'pytanie' not in line.lower()
    )

def process_pdf_and_generate_sections(file_path):
    client = get_genai_client()
    chat = create_chat(client)
    file = upload_pdf(client, file_path)
    first_response = chat.send_message([PROMPT_PARSE, file])
    print(f"Otrzymano pierwszą odpowiedź od modelu:\n{first_response.text[:100]}\n\n")
    n_sections = first_response.text.split("\n")[0].strip().split()[-2]
    try:
        n_sections = int(n_sections)
    except ValueError:
        raise ValueError(f"Nie udało się odczytać liczby sekcji z odpowiedzi: {first_response.text}")
    finished = n_sections <= 10
    further_responses = []
    i = 2
    while not finished:
        for _ in range(5):
            try:
                response = chat.send_message(PROMPT_CONTINUE)
                break
            except (ClientError, ServerError) as e:
                time.sleep(60)
                continue
        else:
            raise RuntimeError("Nie udało się uzyskać odpowiedzi od modelu po 5 próbach.")
        print(f"Otrzymano {i} odpowiedź od modelu:\n{response.text[:100]}\n\n")
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

def generate_basic_flashcards(sections):
    client = get_genai_client()
    print("Rozpoczynam generowanie fiszek...")
    flashcards_responses_og_text = []
    for section in sections:
        for _ in range(5):
            try:
                flashcard_response = generate_flashcards(client, section['text'], PROMPT_GENERATE_FLASHCARDS)
                flashcards_responses_og_text.append(flashcard_response)
                print(f"Udało się wygenerować fiszki dla sekcji {section['id'], section['text'][:100]}")
                break
            except ServerError as e:
                print(f"Błąd podczas generowania fiszek dla sekcji {section['id'], section['text'][:100]}: {e}")
                time.sleep(80)
                continue

    return transform_to_csv(flashcards_responses_og_text)


def generate_synthesis_flashcards(sections):
    print("Rozpoczynam generowanie złożonych fiszek...")
    client = get_genai_client()
    # 1. Połącz wszystkie podsumowania w jeden tekst
    full_summary = "\n\n---\n\n".join([section['summary'] for section in sections])

    synthesis_response_text = []
    for _ in range(5):
        try:
            synthesis_response_text = generate_flashcards(client, full_summary, PROMPT_GENERATE_ADVANCED_FLASHCARDS)
            print("Udało się wygenerować złożone fiszki.")
            break
        except ServerError as e:
            print(f"Błąd podczas generowania złożonych fiszek: {e}")
            time.sleep(80)
            continue
    # synthesis_response_text = generate_flashcards(client, full_summary, PROMPT_GENERATE_ADVANCED_FLASHCARDS)

    return transform_to_csv([synthesis_response_text])

def main():
    sections = process_pdf_and_generate_sections(FILE_PATH)
    time.sleep(20)
    basic_flashcards = generate_basic_flashcards(sections)
    advanced_flashcards = generate_synthesis_flashcards(sections)
    flashcards_combined = basic_flashcards + "\n" + advanced_flashcards

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_flashcards(flashcards_combined, f'flashcards_{timestamp}.csv')


if __name__ == "__main__":
    main()