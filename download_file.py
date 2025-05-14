from bs4 import BeautifulSoup
import os
import time
import requests
import zipfile
import shutil
from urllib.parse import urljoin, urlparse, parse_qs
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- CONFIGURATION ---
BASE_URL      = "https://www.rte-france.com/eco2mix/telecharger-les-indicateurs"
DOWNLOAD_DIR  = "rte_downloads"
EXTRACTED_DIR = "rte_extracted"
ANNUAL_DIR    = os.path.join(EXTRACTED_DIR, "annuel")
CALENDAR_DIR  = os.path.join(EXTRACTED_DIR, "calendrier")
TEMPO_DIR     = os.path.join(EXTRACTED_DIR, "tempo")

EXTENSIONS = ('.csv', '.xls', '.xlsx', '.zip', '.json', '.xml', '.txt')

# Création des dossiers racines
for d in (DOWNLOAD_DIR, EXTRACTED_DIR, ANNUAL_DIR, CALENDAR_DIR, TEMPO_DIR):
    os.makedirs(d, exist_ok=True)

# --- CATÉGORISATION PAR SOUS-CHAÎNE ---
def is_tempo(filename):
    return "tempo" in filename.lower()

def is_annual(filename):
    fn = filename.lower()
    return any(k in fn for k in ("annuel", "annual", "mensuel", "monthly")) or re.search(r"\d{4}", fn)

def is_calendar(filename):
    fn = filename.lower()
    return any(k in fn for k in ("jour", "day", "semaine", "week", "calendar", "calendrier",
                                 "horaire", "hourly", "journalier", "daily"))

def categorize_and_move(path):
    fn = os.path.basename(path)
    if is_tempo(fn):
        dest_dir, tag = TEMPO_DIR, "TEMPO"
    elif is_annual(fn):
        dest_dir, tag = ANNUAL_DIR, "ANNUEL"
    elif is_calendar(fn):
        dest_dir, tag = CALENDAR_DIR, "CALENDRIER"
    else:
        print(f"[NON CLASSÉ] {fn}")
        return
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, fn)
    print(f"[{tag:10}] {fn} → {os.path.basename(dest_dir)}/")
    shutil.move(path, target)

# --- EXTRACTION ZIP ET CLASSIFICATION ---
def extract_and_categorize_zip(zip_path):
    temp_dir = os.path.join(EXTRACTED_DIR, "temp")
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.lower().endswith(('.csv', '.xls', '.xlsx', '.json', '.xml', '.txt')):
                    categorize_and_move(os.path.join(root, f))
    except Exception as e:
        print(f"[Erreur extraction] {zip_path}: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# --- TÉLÉCHARGEMENT D'UN FICHIER ---
def download_file(url):
    """
    Télécharge un fichier dans DOWNLOAD_DIR.
    Si l'URL contient season=XX-XX, on génère un nom dédié pour le calendrier TEMPO.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    if 'season' in qs:
        season = qs['season'][0]  # ex. "15-16"
        fname = f"Calendrier_TEMPO_{season}.zip"
    else:
        fname = os.path.basename(parsed.path)
        if not os.path.splitext(fname)[1]:
            fname += '.zip'

    dest = os.path.join(DOWNLOAD_DIR, fname)
    if os.path.exists(dest):
        print(f"[Existant] {fname}")
        return

    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True, timeout=30)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"[Téléchargé] {fname}")
    except Exception as e:
        print(f"[Échec] {url} → {e}")

# --- RÉCUPÉRATION DE TOUS LES LIENS (INCLUS TEMPO) ---
def fetch_all_links(url, pause=1.0, max_loops=50):
    """
    Charge la page, fait défiler jusqu'à stabilisation des <a>,
    puis renvoie deux sets : les liens TEMPO et les autres fichiers.
    """
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=opts)
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    last_count = 0
    loops = 0
    while loops < max_loops:
        anchors = driver.find_elements(By.TAG_NAME, "a")
        current_count = len(anchors)
        if current_count == last_count:
            break
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        last_count = current_count
        loops += 1

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    tempo_links = set()
    other_links = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(BASE_URL, a['href'])
        if "downloadCalendrierTempo" in href:
            tempo_links.add(href)
        elif href.lower().endswith(EXTENSIONS):
            other_links.add(href)

    return tempo_links, other_links

# --- WORKFLOW PRINCIPAL ---
def main():
    print("⏳ Récupération des liens…")
    tempo_links, other_links = fetch_all_links(BASE_URL)
    print(f"🔵 Calendriers TEMPO détectés : {len(tempo_links)}")
    print(f"🟢 Autres fichiers détectés : {len(other_links)}\n")

    for link in sorted(tempo_links):
        download_file(link)

    for link in sorted(other_links):
        download_file(link)

    for fname in os.listdir(DOWNLOAD_DIR):
        path = os.path.join(DOWNLOAD_DIR, fname)
        if fname.lower().endswith('.zip'):
            extract_and_categorize_zip(path)
        else:
            categorize_and_move(path)

    print("\n✅ Opération terminée : tous les fichiers sont téléchargés et classés.")

if __name__ == "__main__":
    main()

