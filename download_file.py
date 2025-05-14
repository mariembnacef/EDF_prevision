from bs4 import BeautifulSoup
import os
import time
import requests
from urllib.parse import urljoin
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
url = "https://www.rte-france.com/eco2mix/telecharger-les-indicateurs"
download_folder = "rte_downloads"

# Créer le dossier de téléchargement s'il n'existe pas
if not os.path.exists(download_folder):
    os.makedirs(download_folder)
    print(f"Dossier '{download_folder}' créé.")
else:
    print(f"Dossier '{download_folder}' existe déjà.")

def download_file(url, folder):
    """Télécharge un fichier et le sauvegarde dans le dossier spécifié"""
    try:
        # Extraire le nom du fichier depuis l'URL
        filename = os.path.basename(url)
        
        # Si le nom de fichier n'est pas clair, utiliser un nom basé sur l'URL
        if not filename or '?' in filename:
            filename = re.sub(r'[^\w]', '_', url.split('/')[-1].split('?')[0])
            # Ajouter une extension si nécessaire
            if '.' not in filename:
                filename += '.csv'  # Extension par défaut

        filepath = os.path.join(folder, filename)
        
        # Vérifier si le fichier existe déjà
        if os.path.exists(filepath):
            print(f"Le fichier {filename} existe déjà. Passage au suivant.")
            return filepath
        
        print(f"Téléchargement de {filename} depuis {url}")
        
        # Télécharger le fichier avec un User-Agent pour simuler un navigateur
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()  # Vérifier si la requête a réussi
        
        # Vérifier si le contenu est réellement un fichier et non une page HTML
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type and not any(url.lower().endswith(ext) for ext in ['.html', '.htm']):
            print(f"Attention: Le contenu de {url} semble être une page HTML et non un fichier à télécharger.")
            
        # Sauvegarder le fichier
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Fichier sauvegardé: {filepath}")
        return filepath
    
    except Exception as e:
        print(f"Erreur lors du téléchargement de {url}: {e}")
        return None

def download_with_selenium(driver, url, folder):
    """Télécharge un fichier en utilisant Selenium pour gérer les interactions JavaScript"""
    try:
        print(f"Téléchargement avec Selenium: {url}")
        
        # Obtenir le nom du fichier depuis l'URL
        filename = os.path.basename(url)
        if not filename or '?' in filename:
            filename = re.sub(r'[^\w]', '_', url.split('/')[-1].split('?')[0])
            if '.' not in filename:
                filename += '.zip'  # Extension par défaut pour les fichiers RTE
        
        filepath = os.path.join(folder, filename)
        
        # Vérifier si le fichier existe déjà
        if os.path.exists(filepath):
            print(f"Le fichier {filename} existe déjà. Passage au suivant.")
            return filepath
        
        # Naviguer vers l'URL
        driver.get(url)
        
        # Attendre que la page se charge (ajuster le temps d'attente si nécessaire)
        time.sleep(5)
        
        # Pour les sites qui nécessitent de cliquer sur un bouton pour télécharger
        # Rechercher et cliquer sur un bouton de téléchargement si nécessaire
        try:
            # Attendre un court instant pour que la page se charge complètement
            time.sleep(2)
            
            # Essayer de trouver des boutons de téléchargement communs
            download_buttons = driver.find_elements(By.XPATH, 
                "//button[contains(@class, 'download') or contains(text(), 'Download') or contains(text(), 'Télécharger')]")
            
            if download_buttons:
                print(f"Bouton de téléchargement trouvé, tentative de clic...")
                download_buttons[0].click()
                time.sleep(5)  # Attendre que le téléchargement démarre
        except Exception as e:
            print(f"Pas de bouton de téléchargement trouvé ou erreur de clic: {e}")
        
        # Utiliser requests comme méthode alternative si le téléchargement Selenium échoue
        print(f"Tentative de téléchargement direct avec requests pour: {url}")
        return download_file(url, folder)
        
    except Exception as e:
        print(f"Erreur lors du téléchargement avec Selenium de {url}: {e}")
        
        # Essayer de télécharger avec requests en cas d'échec avec Selenium
        print(f"Tentative de téléchargement avec requests comme solution de repli...")
        return download_file(url, folder)

def find_download_links(url):
    """Cherche tous les liens de téléchargement sur la page en utilisant Selenium pour faire défiler"""
    try:
        print(f"Ouverture de la page avec Selenium: {url}")
        
        # Configurer Chrome en mode headless
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Exécuter Chrome en arrière-plan
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Configurer le dossier de téléchargement pour Chrome
        prefs = {
            "download.default_directory": os.path.abspath(download_folder),
            "download.prompt_for_download": False,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Initialiser le navigateur
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Attendre que la page se charge
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Faire défiler la page pour charger tout le contenu
        print("Défilement de la page pour charger tout le contenu...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # Défiler jusqu'en bas
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Attendre le chargement
            time.sleep(2)
            
            # Calculer la nouvelle hauteur
            new_height = driver.execute_script("return document.body.scrollHeight")
            
            # Si la hauteur ne change plus, on a atteint le bas de la page
            if new_height == last_height:
                break
            last_height = new_height
            print("Défilement en cours...")
        
        # Récupérer le contenu HTML complet après défilement
        html_content = driver.page_source
        
        # Fermer le navigateur
        driver.quit()
        
        # Analyser le HTML avec BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Trouver tous les liens (a href) sur la page
        all_links = soup.find_all('a', href=True)
        
        # Filtrer les liens qui semblent être des téléchargements
        download_links = []
        download_extensions = ['.csv', '.xlsx', '.xls', '.zip', '.txt', '.json', '.xml']
        download_keywords = ['download', 'télécharger', 'telecharger', 'export', 'data']
        
        print(f"Nombre total de liens trouvés: {len(all_links)}")
        
        for link in all_links:
            href = link['href']
            link_text = link.get_text().lower().strip()
            
            # Vérifier si le lien contient une extension de fichier à télécharger
            is_download = any(href.lower().endswith(ext) for ext in download_extensions)
            
            # Vérifier si le texte du lien contient un mot-clé de téléchargement
            has_keyword = any(keyword in link_text or keyword in href.lower() for keyword in download_keywords)
            
            if is_download or has_keyword:
                # Convertir en URL absolue si c'est une URL relative
                absolute_url = urljoin(url, href)
                download_links.append(absolute_url)
                print(f"Lien de téléchargement trouvé: {absolute_url}")
        
        return download_links
    
    except Exception as e:
        print(f"Erreur lors de l'analyse de la page {url}: {e}")
        return []

def main():
    print("Initialisation du navigateur Selenium...")
    # Configurer Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Exécuter Chrome en arrière-plan
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Configurer le dossier de téléchargement pour Chrome
    prefs = {
        "download.default_directory": os.path.abspath(download_folder),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": False
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Initialiser le navigateur
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print("Recherche des liens de téléchargement...")
        download_links = find_download_links(url)
        
        print(f"Nombre de liens de téléchargement trouvés: {len(download_links)}")
        
        if not download_links:
            print("Aucun lien de téléchargement trouvé. Vérifiez l'URL ou le contenu de la page.")
            driver.quit()
            return
        
        # Télécharger chaque fichier
        downloaded_files = []
        for i, link in enumerate(download_links):
            print(f"Traitement du lien {i+1}/{len(download_links)}: {link}")
            
            # Essayer d'abord avec Selenium
            filepath = download_with_selenium(driver, link, download_folder)
            
            # Si Selenium échoue, essayer avec requests directement
            if not filepath:
                print(f"Tentative avec method requests pour {link}")
                filepath = download_file(link, download_folder)
                
            if filepath:
                downloaded_files.append(filepath)
            
            # Pause entre les téléchargements pour éviter de surcharger le serveur
            time.sleep(2)
        
        print(f"Téléchargements terminés! {len(downloaded_files)} fichiers téléchargés.")
    
    except Exception as e:
        print(f"Erreur dans le processus principal: {e}")
    
    finally:
        # Toujours fermer le navigateur à la fin
        driver.quit()
        print("Navigateur fermé.")

if __name__ == "__main__":
    main()