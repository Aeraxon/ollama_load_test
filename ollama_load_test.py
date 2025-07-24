import requests
import multiprocessing
import time
import random
import argparse
import signal
import json
import psutil
import threading
from datetime import datetime
from dataclasses import dataclass
from typing import List

@dataclass
class TestResult:
    """Datenklasse für Testergebnisse"""
    users: int
    model: str
    gpu: str
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    avg_ttft: float
    error_rate: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    cpu_usage: float
    memory_usage: float
    test_duration: float
    recommendation: str

class ResultCollector:
    """Sammelt und verwaltet Testergebnisse"""
    def __init__(self):
        self.results = []
        self.lock = threading.Lock()
        
    def add_result(self, result: TestResult):
        with self.lock:
            self.results.append(result)
    
    def get_results(self) -> List[TestResult]:
        with self.lock:
            return self.results.copy()

class SystemMonitor:
    """Überwacht Systemressourcen während des Tests"""
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _monitor_loop(self):
        while self.monitoring:
            try:
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory().percent
                self.cpu_samples.append(cpu)
                self.memory_samples.append(memory)
            except:
                pass
    
    def get_average_cpu(self):
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
    
    def get_average_memory(self):
        return sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0

# Globale Variablen für Ergebnissammlung
response_times = multiprocessing.Manager().list()
ttft_times = multiprocessing.Manager().list()  # Time to First Token
error_count = multiprocessing.Manager().Value('i', 0)
success_count = multiprocessing.Manager().Value('i', 0)

def reset_counters():
    """Setzt die globalen Zähler zurück"""
    global response_times, ttft_times, error_count, success_count
    response_times[:] = []
    ttft_times[:] = []
    error_count.value = 0
    success_count.value = 0

def terminate_processes(processes):
    """Signal-Handler für kontrollierten Abbruch"""
    for p in processes:
        if p.is_alive():
            p.terminate()
    print("Alle Prozesse beendet.")

def load_prompts(file_path):
    """Liest Prompts aus einer Textdatei ein."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def ollama_chat_continuous(model, prompts, user_id, pause_min, pause_max, base_url, test_duration):
    """Simuliert einen Benutzer für eine bestimmte Testdauer"""
    global response_times, ttft_times, error_count, success_count
    
    end_time = time.time() + test_duration
    
    while time.time() < end_time:
        # Zufälligen Prompt auswählen
        prompt = random.choice(prompts)
        
        try:
            start_time = time.time()
            ttft_measured = False
            
            # HTTP-Request an Ollama API mit Streaming für TTFT
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True  # Streaming aktivieren für TTFT-Messung
                },
                timeout=120,
                stream=True
            )
            
            if response.status_code == 200:
                full_response = ""
                first_token_time = None
                
                # Stream-Response verarbeiten
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            
                            # Erstes Token = TTFT
                            if not ttft_measured and 'response' in data and data['response']:
                                first_token_time = time.time() - start_time
                                ttft_times.append(first_token_time)
                                ttft_measured = True
                            
                            # Response sammeln
                            if 'response' in data:
                                full_response += data['response']
                                
                            # Ende der Response
                            if data.get('done', False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
                
                elapsed_time = time.time() - start_time
                response_times.append(elapsed_time)
                success_count.value += 1
                
                # Falls kein Token empfangen wurde, TTFT = Total Time
                if not ttft_measured:
                    ttft_times.append(elapsed_time)
                
                print(f"[User {user_id}] ✓ {elapsed_time:.2f}s (TTFT: {first_token_time:.2f}s) - {prompt[:30]}...")
            else:
                error_count.value += 1
                print(f"[User {user_id}] ✗ HTTP-Fehler {response.status_code}")
                
        except requests.exceptions.Timeout:
            error_count.value += 1
            print(f"[User {user_id}] ✗ Timeout")
        except requests.exceptions.ConnectionError:
            error_count.value += 1
            print(f"[User {user_id}] ✗ Verbindungsfehler")
        except Exception as e:
            error_count.value += 1
            print(f"[User {user_id}] ✗ Fehler: {e}")
        
        # Pause zwischen Requests (nur wenn noch Zeit bleibt)
        if time.time() < end_time:
            pause_time = random.uniform(pause_min, pause_max)
            time.sleep(min(pause_time, end_time - time.time()))

def get_recommendation(avg_time, max_time, error_rate, cpu_usage, avg_ttft):
    """Erstellt eine Empfehlung basierend auf TTFT und anderen Metriken"""
    # Fehlerrate hat höchste Priorität
    if error_rate > 10:
        return "❌ Kritisch"
    elif error_rate > 5:
        return "❌ Überlastet"
    elif error_rate > 2:
        return "⚠️ Instabil"
    # Dann TTFT-basierte Bewertung
    elif avg_ttft > 30:
        return "❌ Inakzeptabel"
    elif avg_ttft > 20:
        return "⚠️ Sehr langsam"
    elif avg_ttft > 10:
        return "⚠️ Langsam"
    elif avg_ttft > 5:
        return "✅ Akzeptabel"
    elif avg_ttft > 2:
        return "✅ Gut"
    else:
        return "✅ Optimal"

def check_ollama_connection(base_url):
    """Prüft ob Ollama erreichbar ist"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def run_load_test(model, prompts, user_count, pause_min, pause_max, test_duration, base_url, gpu_name):
    """Führt einen Load-Test mit einer bestimmten Anzahl von Benutzern durch"""
    reset_counters()
    
    print(f"\n{'='*60}")
    print(f"Test mit {user_count} Benutzern gestartet...")
    print(f"Testdauer: {test_duration/60:.1f} Minuten")
    print(f"{'='*60}")
    
    # System-Monitoring starten
    monitor = SystemMonitor()
    monitor.start_monitoring()
    
    processes = []
    start_time = time.time()
    
    try:
        # Alle Benutzer gleichzeitig starten
        for user_id in range(user_count):
            p = multiprocessing.Process(
                target=ollama_chat_continuous, 
                args=(model, prompts, user_id, pause_min, pause_max, base_url, test_duration)
            )
            p.start()
            processes.append(p)
            
            # Kleine Verzögerung zwischen Starts zur Verteilung
            time.sleep(0.1)
        
        print(f"Alle {user_count} Benutzer gestartet. Warte {test_duration/60:.1f} Minuten...")
        
        # Überwachungsschleife mit Abbruchkriterium
        check_interval = 30  # Prüfe alle 30 Sekunden
        next_check = time.time() + check_interval
        
        while any(p.is_alive() for p in processes):
            time.sleep(1)
            
            # Alle 30 Sekunden Timeout-Rate prüfen
            if time.time() >= next_check:
                total_requests = success_count.value + error_count.value
                if total_requests >= 10:  # Mindestens 10 Requests für aussagekräftige Statistik
                    timeout_rate = (error_count.value / total_requests) * 100
                    print(f"[Zwischenstand] Requests: {total_requests}, Fehlerrate: {timeout_rate:.1f}%")
                    
                    if timeout_rate > 30:
                        print(f"\n⚠️ ABBRUCH: Fehlerrate ({timeout_rate:.1f}%) überschreitet 30%!")
                        print("System ist überlastet - Test wird abgebrochen.")
                        break
                
                next_check = time.time() + check_interval
        
        # Alle Prozesse beenden
        for p in processes:
            if p.is_alive():
                p.terminate()
        
        # Kurz warten, damit Prozesse sauber beenden
        time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nTest abgebrochen...")
        terminate_processes(processes)
        return None
    finally:
        # Sicherstellen, dass alle Prozesse beendet sind
        for p in processes:
            if p.is_alive():
                p.terminate()
    
    # System-Monitoring stoppen
    monitor.stop_monitoring()
    actual_duration = time.time() - start_time
    
    # Ergebnisse auswerten
    times = list(response_times)
    ttft_list = list(ttft_times)
    total_requests = success_count.value + error_count.value
    
    if not times:
        print(f"Keine erfolgreichen Requests in {user_count}-Benutzer-Test!")
        return None
    
    # Empfehlung generieren (jetzt basierend auf TTFT)
    recommendation = get_recommendation(
        sum(times) / len(times),
        max(times),
        (error_count.value / total_requests * 100) if total_requests > 0 else 0,
        monitor.get_average_cpu(),
        sum(ttft_list) / len(ttft_list) if ttft_list else 0
    )
    
    result = TestResult(
        users=user_count,
        model=model,
        gpu=gpu_name,
        avg_response_time=sum(times) / len(times),
        max_response_time=max(times),
        min_response_time=min(times),
        avg_ttft=sum(ttft_list) / len(ttft_list) if ttft_list else 0,
        error_rate=(error_count.value / total_requests * 100) if total_requests > 0 else 0,
        total_requests=total_requests,
        successful_requests=success_count.value,
        failed_requests=error_count.value,
        cpu_usage=monitor.get_average_cpu(),
        memory_usage=monitor.get_average_memory(),
        test_duration=actual_duration,
        recommendation=recommendation
    )
    
    print(f"\nTest abgeschlossen:")
    print(f"  Erfolgreiche Requests: {result.successful_requests}")
    print(f"  Fehlgeschlagene Requests: {result.failed_requests}")
    print(f"  Durchschnittliche Antwortzeit: {result.avg_response_time:.2f}s")
    print(f"  Durchschnittliche TTFT: {result.avg_ttft:.2f}s")
    print(f"  Maximale Antwortzeit: {result.max_response_time:.2f}s")
    print(f"  Fehlerrate: {result.error_rate:.1f}%")
    print(f"  CPU-Auslastung: {result.cpu_usage:.1f}%")
    
    return result

def print_results_table(results: List[TestResult]):
    """Gibt die Ergebnistabelle aus"""
    if not results:
        print("Keine Ergebnisse zum Anzeigen.")
        return
    
    print(f"\n{'='*138}")
    print("LOAD TEST ERGEBNISSE")
    print(f"{'='*138}")
    
    # Header
    print(f"{'Benutzer':<8} {'Modell':<15} {'GPU':<12} {'Avg. Zeit':<10} {'TTFT':<8} {'Max. Zeit':<10} {'Min. Zeit':<10} {'Fehlerrate':<11} {'CPU %':<8} {'Memory %':<10} {'Requests':<10} {'Empfehlung':<12}")
    print(f"{'-'*8} {'-'*15} {'-'*12} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*11} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
    
    # Datenzeilen
    for result in results:
        print(f"{result.users:<8} {result.model:<15} {result.gpu:<12} {result.avg_response_time:<10.2f} {result.avg_ttft:<8.2f} {result.max_response_time:<10.2f} {result.min_response_time:<10.2f} {result.error_rate:<11.1f} {result.cpu_usage:<8.1f} {result.memory_usage:<10.1f} {result.total_requests:<10} {result.recommendation:<12}")
    
    print(f"{'-'*138}")

def save_results_to_file(results: List[TestResult], filename: str):
    """Speichert Ergebnisse in eine CSV-Datei"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # CSV-Header
            f.write("Benutzer,Modell,GPU,Avg_Antwortzeit,Avg_TTFT,Max_Antwortzeit,Min_Antwortzeit,Fehlerrate,CPU_Prozent,Memory_Prozent,Total_Requests,Erfolgreiche_Requests,Fehlgeschlagene_Requests,Testdauer,Empfehlung\n")
            
            # Datenzeilen
            for result in results:
                f.write(f"{result.users},{result.model},{result.gpu},{result.avg_response_time:.3f},{result.avg_ttft:.3f},{result.max_response_time:.3f},{result.min_response_time:.3f},{result.error_rate:.2f},{result.cpu_usage:.2f},{result.memory_usage:.2f},{result.total_requests},{result.successful_requests},{result.failed_requests},{result.test_duration:.1f},{result.recommendation}\n")
        
        print(f"\nErgebnisse gespeichert in: {filename}")
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")

def main():
    parser = argparse.ArgumentParser(description="Schrittweises Load Testing für Ollama")
    parser.add_argument("--prompts", type=str, required=True, 
                       help="Pfad zur Prompts-Datei")
    parser.add_argument("--users", type=int, required=True, 
                       help="Maximale Anzahl der Benutzer (wird schrittweise erreicht)")
    parser.add_argument("--model", type=str, required=True, 
                       help="Ollama-Modell(e), kommagetrennt für mehrere Modelle")
    parser.add_argument("--gpu", type=str, default="Unknown", 
                       help="GPU-Bezeichnung für Dokumentation (Standard: Unknown)")
    parser.add_argument("--pause-min", type=float, default=3.0, 
                       help="Minimale Pause zwischen Nachrichten in Sekunden (Standard: 3.0)")
    parser.add_argument("--pause-max", type=float, default=30.0, 
                       help="Maximale Pause zwischen Nachrichten in Sekunden (Standard: 30.0)")
    parser.add_argument("--step-size", type=int, default=5, 
                       help="Schrittgröße für Benutzererhöhung (Standard: 5)")
    parser.add_argument("--test-duration", type=int, default=300, 
                       help="Testdauer pro Schritt in Sekunden (Standard: 300 = 5 Minuten)")
    parser.add_argument("--host", type=str, default="127.0.0.1:11434", 
                       help="Ollama Host und Port (Standard: 127.0.0.1:11434)")
    parser.add_argument("--output", type=str, default=None, 
                       help="Dateiname für CSV-Export (optional)")
    
    args = parser.parse_args()
    
    # Modelle aus kommagetrenntner Liste extrahieren
    models = [model.strip() for model in args.model.split(',') if model.strip()]
    
    if not models:
        print("Fehler: Keine gültigen Modelle angegeben!")
        return
    
    # Base URL aus Host-Parameter erstellen
    if not args.host.startswith(('http://', 'https://')):
        base_url = f"http://{args.host}"
    else:
        base_url = args.host
    
    # Validierung
    if args.pause_min > args.pause_max:
        print("Fehler: pause-min darf nicht größer als pause-max sein!")
        return
    
    if args.users <= 0 or args.step_size <= 0:
        print("Fehler: users und step-size müssen größer als 0 sein!")
        return
    
    # Ollama-Verbindung prüfen
    print(f"Prüfe Verbindung zu Ollama ({base_url})...")
    if not check_ollama_connection(base_url):
        print(f"Fehler: Kann nicht zu Ollama unter {base_url} verbinden!")
        print("Stelle sicher, dass Ollama läuft: ollama serve")
        return
    
    # Prompts laden
    try:
        prompts = load_prompts(args.prompts)
        print(f"✓ {len(prompts)} Prompts aus {args.prompts} geladen")
    except FileNotFoundError:
        print(f"Fehler: Prompts-Datei {args.prompts} nicht gefunden!")
        return
    
    if len(prompts) == 0:
        print("Fehler: Keine Prompts in der Datei gefunden!")
        return
    
    # Test-Parameter anzeigen
    print(f"\nSTARTE SCHRITTWEISES LOAD TESTING")
    print(f"Modelle: {', '.join(models)}")
    print(f"GPU: {args.gpu}")
    print(f"Maximale Benutzer: {args.users}")
    print(f"Schrittgröße: {args.step_size}")
    print(f"Testdauer pro Schritt: {args.test_duration/60:.1f} Minuten")
    print(f"Pausenzeiten: {args.pause_min}-{args.pause_max} Sekunden")
    print(f"Host: {base_url}")
    
    # Schrittweise Tests durchführen
    results = []
    user_steps = list(range(args.step_size, args.users + 1, args.step_size))
    
    # Falls die maximale Anzahl nicht durch step_size teilbar ist, hinzufügen
    if args.users not in user_steps:
        user_steps.append(args.users)
    
    total_steps = len(user_steps) * len(models)
    estimated_total_time = total_steps * args.test_duration / 60
    
    print(f"Geplante Schritte: {user_steps}")
    print(f"Geschätzte Gesamtdauer: {estimated_total_time:.1f} Minuten")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        step_counter = 0
        
        # Für jedes Modell alle Benutzer-Schritte durchführen
        for model in models:
            print(f"\n{'='*80}")
            print(f"TESTE MODELL: {model}")
            print(f"{'='*80}")
            
            for user_count in user_steps:
                step_counter += 1
                print(f"\n[Schritt {step_counter}/{total_steps}] Teste {user_count} Benutzer mit {model}...")
                
                result = run_load_test(
                    model, prompts, user_count, 
                    args.pause_min, args.pause_max, 
                    args.test_duration, base_url, args.gpu
                )
                
                if result:
                    results.append(result)
                
                # Kurze Pause zwischen Tests
                if step_counter < total_steps:
                    print("Pause zwischen Tests (10 Sekunden)...")
                    time.sleep(10)
        
        # Ergebnisse anzeigen
        print_results_table(results)
        
        # CSV-Export falls gewünscht
        if args.output:
            save_results_to_file(results, args.output)
        else:
            # Automatischer Dateiname mit bereinigten Modellnamen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Modellnamen für Dateiname bereinigen (/, : durch _ ersetzen)
            clean_models = [model.replace('/', '_').replace(':', '_') for model in models]
            models_str = "_".join(clean_models)[:50]  # Begrenzen für Dateinamen
            filename = f"ollama_load_test_{models_str}_{timestamp}.csv"
            save_results_to_file(results, filename)
        
        print(f"\nLoad Test abgeschlossen um {datetime.now().strftime('%H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\nLoad Test abgebrochen!")
        if results:
            print("Bisherige Ergebnisse:")
            print_results_table(results)

if __name__ == "__main__":
    main()
