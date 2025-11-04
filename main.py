import json
import os
import re
import time
import datetime

from google import genai
from google.genai.errors import ClientError, ServerError
from config import API_KEY


FILE_PATH = r''

OUTPUT_DIR = 'output'
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 60

PROMPT_PARSE = """
W załączniku znajduje się wykład z zajęć w postaci pliku PDF.
Twoim zadaniem jest:
- Przeczytać cały dokument, łącznie z tekstem na zdjęciach (użyj OCR).
- Na podstawie treści i tematyki oddzielić wykład na logiczne części, gdzie każda ma maksymalnie 1200 słów i łącznie obejmą materiał z CAŁEGO dokumentu.
- W pierwszej linii odpowiedzi, którą wygenerujesz, umieścić informację na ile sekcji podzieliłeś wykład, w formacie "Podzielono na X sekcji". 
Chodzi o łączną liczbę sekcji, która pozwoli optymalnie objąć materiał z całego dokumentu, a nie ile wygenerujesz w tej odpowiedzi (maksymalnie 10). 
- Dla każdego slajdu zawierającego obraz (mapę, wykres, diagram), krótko opisz, co ten obraz przedstawia, np. "Slajd 64 pokazuje diagram semiwariogramu z zaznaczonymi komponentami: nugget, sill, range".
- Dla każdej sekcji wygenerować podsumowanie informacji zawartych w tej części (do 800 słów) oraz pełny tekst.
- Zinterpretuj także wzory matematyczne i zapisz je w formacie MathJax (czyli np. \\( a^2 + b^2 = c^2 \\) ). Jeśli wzory są nieczytelne (np. na skanach), spróbuj odtworzyć ich znaczenie na podstawie kontekstu. Zadbaj o poprawną składnię MathJax, nadającą się do użycia np. w Anki.
- Dla każdej sekcji wygenerować podsumowanie, pełny tekst oraz listę kluczowych koncepcji.

Zwracaj sekcje w następującym formacie:

Section: <numer>
Title: <tytuł sekcji>
Slides: <numery slajdów>
Summary:
<podsumowanie sekcji>
KeyConcepts:
- Lista kluczowych, fundamentalnych definicji, wzorów i zasad z tej sekcji. Skup się na wiedzy, która jest niezbędna do dalszego zrozumienia tematu. Np. "Wzór na dystans Euklidesowy", "Definicja efektu samorodkowego (nugget)", "Zasada działania algorytmu KNN".
FullText:
<pełny tekst sekcji>

Jeśli podział będzie dłuższy niż 10 sekcji, zatrzymaj się na sekcji 10 – późniejsze sekcje wygeneruję w osobnym zapytaniu.
"""

PROMPT_CONTINUE = """
Kontynuuj z następnymi sekcjami, które nie zostały stworzone w poprzedniej wiadomości (analogiczny format jak poprzednio). 
"""

PROMPT_GENERATE_FLASHCARDS = """
Wciel się w rolę doświadczonego dydaktyka i eksperta od nauk kognitytywnych. Twoim zadaniem jest stworzenie talii fiszek na podstawie dostarczonego tekstu. Celem jest stworzenie narzędzi do aktywnego przypominania, które maksymalizują głębokie zrozumienie i długoterminowe zapamiętywanie, a nie tylko testują powierzchowną znajomość samego tekstu źródłowego.

- Struktura Danych Wejściowych
- Otrzymasz tekst, który może zawierać dwie części:
  1. KeyConcepts: Lista kluczowych, fundamentalnych definicji, wzorów i zasad z tej sekcji, które student MUSI zapamiętać.
  2. FullText: Pełny kontekst wyjaśniający te koncepcje.

- Główne Zadania
1.  Stwórz fiszki dla KAŻDEGO elementu z listy KeyConcepts. Mają to być precyzyjne pytania testujące znajomość definicji, wzoru lub podstawowej zasady.
2.  Przeanalizuj FullText i stwórz dodatkowe fiszki, które testują ZROZUMIENIE. Nie powtarzaj pytań z KeyConcepts, ale rozszerzaj je, pytając o ich znaczenie i zastosowanie.

- Zasady Tworzenia Wysokiej Jakości Fiszki

A. Styl i Struktura Pytania:
- Pytanie powinno być jak najkrótsze, proste, jasne i skoncentrowane na najważniejszych informacjach. Musi wymagać aktywnego przypomnienia sobie odpowiedzi.
- Odpowiedź powinna być zwięzła, ale zawierać kompletne wyjaśnienie.
- Priorytetyzuj pytania typu "Jak działa...?", "Dlaczego...?", "Jaki jest cel...?", "Jaki problem rozwiązuje...?" oraz "Czym się różni...?".
- Jeśli w tekście jest przykład liczbowy lub studium przypadku, stwórz fiszkę o OGÓLNEJ ZASADZIE, którą ten przykład ilustruje.
- Jeśli w tekście znajduje się lista, stwórz OSOBNE pytanie dla każdego ważnego elementu, pytając o jego funkcję lub cel.

B. Czego Należy Unikać:
- UNIKAJ ZŁYCH PYTAŃ: Pytanie nie może sugerować ani zawierać odpowiedzi.
  - PRZYKŁAD ZŁEJ PRAKTYKI: ŹLE: "Jaka metoda poprawia jakość estymatorów z drzewa, jest metodą samowsporną i nazywana jest 'zbiorową mądrością'?" ; "Bagging"
- UNIKAJ PYTAŃ O LISTY: Nie twórz pytań w stylu "Wymień X...", "Podaj Y rodzajów...".
- UNIKAJ FAKTÓW TRYWIALNYCH: Nie pytaj o konkretne wartości liczbowe (np. R^2=0.84), parametry ((0, 0, 500)), nazwy własne czy wyniki pochodzące z pojedynczych, ilustracyjnych przykładów w tekście.
- UNIKAJ INFORMACJI NIEISTOTNYCH: Ignoruj informacje, które nie mają wartości merytorycznej z perspektywy studenta (dane organizacyjne, nazwiska niezwiązane z kluczową teorią, detale zadań projektowych).

Przykłady Dobrej i Złej Praktyki
- Przykład 1 (Listy):
  - ŹLE: Wymień 4 skale pomiarowe atrybutów.
  - LEPIEJ: Czym charakteryzuje się skala pomiarowa nominalna i podaj jej przykład. ORAZ Jaka jest kluczowa różnica między skalą interwałową a ilorazową?

- Przykład 2 (Dane z przykładów):
  - ŹLE: Jaka była wartość R^2 dla modelu ratingu w przykładzie Cereals?
  - LEPIEJ: Co interpretujemy za pomocą współczynnika determinacji R² w modelu regresji i co oznacza jego wysoka wartość?

- Przykład 3 (Fakty fundamentalne vs trywialne):
  - ŹLE: Jakie RMSE uzyskano dla Modelu 0 (średnia)? (Trywialny, nietransferowalny fakt)
  - DOBRZE: Jaki jest wzór na standaryzację danych i dlaczego jest ona ważnym krokiem w PCA? (Fundamentalny, transferowalny fakt)

Wymagania Techniczne i Formatowanie
- Język: Fiszki muszą być w języku polskim.
- Format Wyjściowy: Fiszki mają być wygenerowane w formie CSV z dokładnie dwiema kolumnami: `pytanie` i `odpowiedź`.
- Separator: Separatorem kolumn musi być średnik (;).
- KRYTYCZNA ZASADA: Treść pól 'pytanie' i 'odpowiedź' NIGDY nie może zawierać średników (;), nawet jeśli jest to gramatycznie poprawne. Każdy średnik uszkodzi plik. Jeśli odpowiedź wymaga wyliczenia, ZAWSZE używaj tagów HTML<ul>i<li>. Na przykład:<ul><li>Punkt pierwszy</li><li>Punkt drugi</li></ul>.
- Wzory Matematyczne: Wszystkie wzory muszą być umieszczone w formacie MathJax (np. \( a^2 + b^2 = c^2 \) ).
- Formatowanie Tekstu: Do formatowania tekstu, takiego jak pogrubienie lub kursywa, używaj wyłącznie tagów HTML (np. <b>pogrubienie</b>, <i>kursywa</i>). Nie używaj formatowania Markdown, ponieważ nie jest ono obsługiwane.
- Bloki Kodu: Wszystkie fragmenty kodu MUSZĄ być umieszczone wewnątrz tagów<pre><code>...</code></pre>. Nie używaj potrójnych backticków (```).
- Pełne pokrycie: Fiszki powinny wyczerpująco opisywać zadany tekst, jednocześnie nie powtarzając się.

Teraz, działając jako ekspert dydaktyczny, stwórz fiszki na podstawie poniższego tekstu. W swojej odpowiedzi nie umieszczaj żadnych dodatkowych informacji, tylko i wyłącznie tabelę z fiszkami w formacie CSV.
"""

PROMPT_GENERATE_ADVANCED_FLASHCARDS = """
- Rola i Główny Cel
- Twoim zadaniem jest stworzenie ZAAWANSOWANEJ talii fiszek z udostępnionego tekstu, który jest podsumowaniem całego wykładu. W przeciwieństwie do fiszek podstawowych, które testują znajomość pojedynczych koncepcji, te fiszki mają testować głębsze zrozumienie, zdolność do syntezy i porównywania koncepcji.

Instrukcje Ogólne
- Pytanie powinno być zwięzłe i prowokować do myślenia, a nie tylko odtwarzania faktów.
- Odpowiedź powinna być wyczerpująca, dobrze ustrukturyzowana i jasno wyjaśniać zagadnienie.
- Unikaj pytań, na które odpowiedź to jedno słowo.
- Jeśli pytanie dotyczy elementu wizualnego (mapy, wykresu), zaznacz to w pytaniu i podaj numer slajdu, np. "[PYTANIE WIZUALNE, slajd 57]".

- Kategorie Pytań do Wygenerowania
- Wygeneruj fiszki z następujących kategorii, aby zapewnić różnorodność i głębię nauki:
  1. Pytania Porównawcze/Syntetyzujące:
     - Mają na celu porównanie dwóch lub więcej koncepcji, metod lub narzędzi.
     - Powinny pytać o podobieństwa, różnice, wady i zalety.
     - Przykład: "Porównaj mechanizm działania Random Forest i Gradient Boosting, wskazując na kluczowe różnice w budowie drzew i celu uczenia."

  2. Pytania Koncepcyjne/Wyjaśniające ("Dlaczego?"):
     - Mają na celu sprawdzenie zrozumienia przyczyn i mechanizmów.
     - Powinny zaczynać się od "Wyjaśnij, dlaczego...", "W jaki sposób...", "Jaki problem rozwiązuje...".
     - Przykład: "Wyjaśnij, dlaczego modele oparte na drzewach decyzyjnych nie potrafią ekstrapolować i jakie ma to praktyczne konsekwencje w analizie danych przestrzennych."

  3. Pytania Praktyczne/Zastosowanie (Oparte na przykładach z wykładu):
     - Mają na celu połączenie teorii z konkretnymi przykładami omówionymi w materiale, jeśli takie występują
     - Przykład: "Opisz, jak w przykładzie z danymi kolarskimi wykorzystano regresję logistyczną i macierz błędów do zbudowania i oceny klasyfikatora wysiłku tlenowego/beztlenowego."

- Przykład
Pytanie;Odpowiedź
Porównaj kryteria informacyjne AIC i BIC.;Oba kryteria równoważą dopasowanie modelu \( ln(L) )) i jego złożoność (liczbę parametrów \( k \)). Różnią się karą: w AIC jest stała (2), a w BIC zależy od liczby danych (\( ln(n) \)). Sprawia to, że BIC jest bardziej "krytyczny" i preferuje prostsze modele, podczas gdy AIC jest lepszy w wyborze modelu o najlepszych zdolnościach predykcyjnych.
Wyjaśnij, dlaczego w algebrze map konieczne jest skalowanie wartości przed użyciem niektórych funkcji odległości?;Ponieważ funkcje odległości, takie jak dystans Euklidesowy, są wrażliwe na skalę zmiennych. Jeśli jedna cecha (np. współrzędna geograficzna) ma znacznie większy zakres wartości niż inna, zdominuje ona obliczenia odległości, prowadząc do błędnych wyników. Skalowanie (np. normalizacja) sprowadza wszystkie cechy do wspólnego zakresu, zapewniając im równy wkład.

- Wymagania Techniczne i Formatowanie
- Język: Fiszki muszą być w języku polskim.
- Format Wyjściowy: Fiszki mają być wygenerowane w formie CSV z dokładnie dwiema kolumnami: `pytanie` i `odpowiedź`.
- Separator: Separatorem kolumn musi być średnik (;).
- Wzory Matematyczne: Wszystkie wzory muszą być umieszczone w formacie MathJax (np. \( a^2 + b^2 = c^2 \) ).
- Formatowanie Tekstu: Do formatowania tekstu, takiego jak pogrubienie lub kursywa, używaj wyłącznie tagów HTML (np. <b>pogrubienie</b>, <i>kursywa</i>). Nie używaj formatowania Markdown.
- KRYTYCZNA ZASADA: Treść pól 'pytanie' i 'odpowiedź' NIGDY nie może zawierać średników (;), nawet jeśli jest to gramatycznie poprawne. Każdy średnik uszkodzi plik. Jeśli odpowiedź wymaga wyliczenia, ZAWSZE używaj tagów HTML<ul> i <li>. Na przykład:<ul><li>Punkt pierwszy</li><li>Punkt drugi</li></ul>.
- Bloki Kodu: Wszystkie fragmenty kodu MUSZĄ być umieszczone wewnątrz tagów<pre><code>...</code></pre>. Nie używaj potrójnych backticków (```).

Teraz stwórz zaawansowane fiszki na podstawie poniższego tekstu. W swojej odpowiedzi nie umieszczaj żadnych dodatkowych informacji, tylko i wyłącznie tabelę z fiszkami w formacie CSV.
"""


def get_genai_client():
    return genai.Client(api_key=API_KEY)

def upload_pdf(client, file_path):
    print(f"Wysyłanie pliku: {file_path}...")
    return client.files.upload(file=file_path)

def create_chat(client, model="gemini-2.5-flash-preview-05-20"):
    return client.chats.create(model=model)

def parse_slide_range(slide_str):
    if not slide_str: return []
    slides = []
    for part in slide_str.split(','):
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                slides.extend(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                slides.append(int(part))
            except ValueError:
                continue
    return slides


def parse_sections(text):
    sections = []
    raw_section_blocks = re.split(r'\nSection:\s*(\d+)', text)

    for i in range(1, len(raw_section_blocks), 2):
        sec_num = int(raw_section_blocks[i])
        content = raw_section_blocks[i + 1]

        title_match = re.search(r'Title:\s*(.*?)\n', content, re.DOTALL)
        slides_match = re.search(r'Slides:\s*(.*?)\n', content, re.DOTALL)
        summary_match = re.search(r'Summary:\s*(.*?)\nKeyConcepts:', content, re.DOTALL)
        keyconcepts_match = re.search(r'KeyConcepts:\s*(.*?)\nFullText:', content, re.DOTALL)
        fulltext_match = re.search(r'FullText:\s*(.*)', content, re.DOTALL)

        sections.append({
            "id": sec_num,
            "title": title_match.group(1).strip() if title_match else "",
            "slides": parse_slide_range(slides_match.group(1).strip() if slides_match else ""),
            "summary": summary_match.group(1).strip() if summary_match else "",
            "key_concepts": keyconcepts_match.group(1).strip() if keyconcepts_match else "",
            "full_text": fulltext_match.group(1).strip() if fulltext_match else ""
        })
    return sections


def save_sections(sections):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filepath = os.path.join(OUTPUT_DIR, f'sections_{timestamp}.json')
    with open(filepath, 'w', encoding='utf-8-sig') as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)
    print(f"Zapisano sekcje w pliku: {filepath}")


def save_flashcards(response, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8-sig') as f:
        f.write(response + '\n')
    print(f"Zapisano fiszki w pliku: {filepath}")


def transform_to_csv(responses):
    all_lines = []
    for response in responses:
        if not response:
            continue

        lines = response.strip().split('\n')
        for line in lines:
            # Ignoruj puste linie lub linie z nagłówkiem, jeśli model je zwróci
            if ';' in line and 'pytanie;odpowiedź' not in line.lower():
                # Każda linia jest najpierw korygowana przez post-processing
                corrected_line = post_process_flashcard_line(line)
                all_lines.append(corrected_line)

    return '\n'.join(all_lines)


def process_pdf_and_generate_sections(client, file_path):
    chat = create_chat(client)
    file = upload_pdf(client, file_path)

    print("Wysyłanie początkowego promptu do analizy PDF...")
    first_response = None
    for attempt in range(MAX_RETRIES):
        try:
            first_response = chat.send_message([PROMPT_PARSE, file])
            if not first_response or not getattr(first_response, "text", None):
                raise ServerError("Otrzymano pustą odpowiedź od modelu.")
            break
        except (ClientError, ServerError) as e:
            print(f"Błąd API przy pierwszym zapytaniu (próba {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise RuntimeError("Nie udało się uzyskać początkowej odpowiedzi od modelu.")

    print(f"Otrzymano pierwszą odpowiedź:\n{first_response.text[:150]}...\n")

    n_sections_match = re.search(r'Podzielono na (\d+) sekcji', first_response.text)
    if not n_sections_match:
        raise ValueError(f"Nie udało się odczytać liczby sekcji z odpowiedzi: {first_response.text}")
    n_sections = int(n_sections_match.group(1))
    print(f"Dokument został podzielony na {n_sections} sekcji.")

    all_responses = [first_response.text]
    num_processed_sections = len(re.findall(r'\nSection:', first_response.text))

    while num_processed_sections < n_sections:
        print(f"Pobieranie kolejnych sekcji (pobrano {num_processed_sections}/{n_sections})...")
        response = None
        for attempt in range(MAX_RETRIES):
            try:
                response = chat.send_message(PROMPT_CONTINUE)
                if not response or not getattr(response, "text", None):
                    raise ServerError(f"Otrzymano pustą odpowiedź od modelu przy kontynuacji.")
                break
            except (ClientError, ServerError) as e:
                print(f"Błąd API przy kontynuacji (próba {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print("Nie udało się pobrać dalszych sekcji. Przetwarzam dotychczasowe dane.")
                    response = None
                    break

        if response is None: break

        print(f"Otrzymano kolejną odpowiedź:\n{response.text[:150]}...\n")
        all_responses.append(response.text)
        num_processed_sections += len(re.findall(r'\nSection:', response.text))

    sections = []
    for text in all_responses:
        sections.extend(parse_sections(text))
    save_sections(sections)
    return sections


def _generate_flashcards_with_retry(client, text, prompt):
    """Wewnętrzna funkcja pomocnicza z logiką ponawiania prób."""
    for attempt in range(MAX_RETRIES):
        try:
            flashcard_response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt + text,
            )
            time.sleep(5)
            if not flashcard_response or not getattr(flashcard_response, "text", None):
                print(f"Błąd treści (próba {attempt + 1}/{MAX_RETRIES}): Otrzymano pustą odpowiedź od modelu.")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                else:
                    return ""

            response_text = flashcard_response.text
            if has_acceptable_format(response_text):
                return response_text
            else:
                print(f"Błąd formatu (próba {attempt + 1}/{MAX_RETRIES}): Otrzymano nieprawidłowy format CSV.")
                print(f"Fragment błędnej odpowiedzi: {response_text[:200]}...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                else:
                    print(f"Nie udało się uzyskać poprawnego formatu po {MAX_RETRIES} próbach.")
                    return ""

        except (ClientError, ServerError) as e:
            print(f"Błąd API (próba {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Ponowna próba za {RETRY_DELAY_SECONDS} sekund...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"Nie udało się wygenerować fiszek po {MAX_RETRIES} próbach API dla tekstu: {text[:100]}...")
                return ""
    return ""


def generate_basic_flashcards(client, sections):
    print("\nRozpoczynam generowanie fiszek podstawowych...")
    flashcards_responses = []
    for i, section in enumerate(sections):
        if not section.get('key_concepts') and not section.get('full_text'):
            print(f"Pominięto sekcję {section['id']}, ponieważ jest pusta.")
            continue

        input_text = (
            f"KeyConcepts:\n{section['key_concepts']}\n\n"
            f"---\n\n"
            f"FullText:\n{section['full_text']}"
        )

        print(f"Generowanie fiszek dla sekcji {section['id']}: '{section['title']}' ({i + 1}/{len(sections)})...")
        response = _generate_flashcards_with_retry(client, input_text, PROMPT_GENERATE_FLASHCARDS)
        flashcards_responses.append(response)
        if response:
            print(f"-> Sukces dla sekcji {section['id']}.")
        else:
            print(f"-> NIEPOWODZENIE dla sekcji {section['id']}.")

    return transform_to_csv(flashcards_responses)


def generate_synthesis_flashcards(client, sections):
    print("\nRozpoczynam generowanie fiszek zaawansowanych (syntetyzujących)...")

    CHUNK_SIZE = 7
    all_synthesis_responses = []

    summaries = [s['summary'] for s in sections if s.get('summary')]

    for i in range(0, len(summaries), CHUNK_SIZE):
        chunk = summaries[i:i + CHUNK_SIZE]
        full_summary_chunk = "\n\n---\n\n".join(chunk)

        if not full_summary_chunk.strip():
            continue

        print(f"Generowanie fiszek zaawansowanych dla paczki {i // CHUNK_SIZE + 1}...")
        response = _generate_flashcards_with_retry(client, full_summary_chunk, PROMPT_GENERATE_ADVANCED_FLASHCARDS)

        if response:
            all_synthesis_responses.append(response)
            print(f"-> Sukces dla paczki {i // CHUNK_SIZE + 1}.")
        else:
            print(f"-> NIEPOWODZENIE dla paczki {i // CHUNK_SIZE + 1}.")

    if not all_synthesis_responses:
        print("Nie udało się wygenerować żadnych fiszek zaawansowanych.")
        return ""

    return transform_to_csv(all_synthesis_responses)


def has_acceptable_format(text: str, max_invalid_starts: int = 4) -> bool:
    if not text or not text.strip():
        return False

    lines = text.strip().split('\n')
    if not lines:
        return False

    answer_without_question_count = 0
    for line in lines:
        if line.strip().lower().startswith('odpowiedź;'):
            answer_without_question_count += 1

    if answer_without_question_count > max_invalid_starts:
        print(
            f"Walidacja nieudana: znaleziono {answer_without_question_count} linii zaczynających się od 'odpowiedź;' (limit: {max_invalid_starts}).")
        return False
    return True


def post_process_flashcard_line(line: str) -> str:
    parts = line.split(';')

    question = ""
    answer = ""

    if len(parts) == 1:
        question = parts[0].strip()
    elif len(parts) == 2:
        question = parts[0].strip()
        answer = parts[1].strip()
    elif len(parts) > 2:
        question = parts[0].strip()
        answer = ";".join(parts[1:]).strip()

    fields_to_process = [question, answer]
    processed_fields = []

    for field in fields_to_process:
        if not isinstance(field, str):
            field = str(field)

        # **text** -> <b>text</b>
        field = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', field)
        # *text* -> <i>text</i>
        field = re.sub(r'(?<!\*)\*(\S(.*?)\S)\*(?!\*)', r'<i>\1</i>', field)
        # `text` -> <code>text</code>
        field = re.sub(r'`(.*?)`', r'<code>\1</code>', field)
        processed_fields.append(field)

    question, answer = processed_fields
    # czyszczenie zduplikowanych tagów <code> w <pre>
    answer = answer.replace('<pre><code><code>', '<pre><code>').replace('</code></code></pre>', '</code></pre>')
    return f"{question};{answer}"

def main():
    client = get_genai_client()

    # Krok 1: Przetwarzanie PDF i podział na sekcje
    sections = process_pdf_and_generate_sections(client, FILE_PATH)
    if not sections:
        print("Nie udało się wygenerować żadnych sekcji. Zakończono program.")
        return

    # Krok 2: Generowanie fiszek podstawowych
    basic_flashcards_csv = generate_basic_flashcards(client, sections)

    # Krok 3: Generowanie fiszek zaawansowanych
    advanced_flashcards_csv = generate_synthesis_flashcards(client, sections)

    # Krok 4: Połączenie i zapisanie wyników
    flashcards_combined = basic_flashcards_csv
    if advanced_flashcards_csv:
        if basic_flashcards_csv:
            flashcards_combined += "\n"
        flashcards_combined += advanced_flashcards_csv

    if flashcards_combined.strip():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        save_flashcards(flashcards_combined, f'flashcards_{timestamp}.csv')
        print("\nZakończono pomyślnie! Fiszki zostały zapisane.")
    else:
        print("\nNie udało się wygenerować żadnych fiszek.")


if __name__ == "__main__":
    main()