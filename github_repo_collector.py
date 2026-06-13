"""
GitHub Repository Collector — Game Development in C++
======================================================
Busca repositórios públicos de desenvolvimento de games em C++ via GitHub API
e salva um dataset CSV com métricas de engenharia de software.

Requisitos:
    pip install requests pandas python-dotenv

Uso:
    1. Defina GITHUB_TOKEN no ambiente ou em arquivo .env
    2. Execute: python github_repo_collector.py
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ── Configuração ────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")          # Personal Access Token
OUTPUT_FILE  = "cpp_game_repos.csv"
TARGET_REPOS = 50                                       # quantos repositórios buscar

SEARCH_QUERIES = [
    "game engine language:C++ stars:>50",
    "game development C++ topic:game",
    "opengl game language:C++ stars:>30",
    "vulkan game language:C++ stars:>20",
    "2d game engine language:C++ stars:>20",
    "3d game language:C++ topic:gamedev",
    "sfml game language:C++ stars:>10",
    "sdl2 game language:C++ stars:>10",
]

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"
else:
    logging.warning("GITHUB_TOKEN não definido — limite de 60 req/h se aplica.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Funções auxiliares ───────────────────────────────────────────────────────

def search_repos(query: str, per_page: int = 30, max_pages: int = 2) -> list[dict]:
    """Busca repositórios usando a Search API do GitHub."""
    results = []
    for page in range(1, max_pages + 1):
        url = "https://api.github.com/search/repositories"
        params = {"q": query, "per_page": per_page, "page": page, "sort": "stars", "order": "desc"}
        resp = _get(url, params)
        if resp is None:
            break
        items = resp.get("items", [])
        results.extend(items)
        if len(items) < per_page:
            break
        time.sleep(1)   # respeita rate limit
    return results


def get_repo_details(owner: str, repo: str) -> dict | None:
    """Busca detalhes adicionais de um repositório específico."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    return _get(url)


def get_contributors_count(owner: str, repo: str) -> int:
    """Conta o número de contribuidores de um repositório."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    params = {"per_page": 1, "anon": "true"}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    if resp.status_code == 200:
        # o total vem no cabeçalho Link da última página
        link = resp.headers.get("Link", "")
        if 'rel="last"' in link:
            import re
            match = re.search(r'page=(\d+)>; rel="last"', link)
            if match:
                return int(match.group(1))
        return len(resp.json())
    return 0


def get_languages(owner: str, repo: str) -> dict:
    """Retorna o breakdown de linguagens (em bytes) do repositório."""
    url = f"https://api.github.com/repos/{owner}/{repo}/languages"
    result = _get(url)
    return result if result else {}


def get_commits_count(owner: str, repo: str) -> int:
    """Estima o total de commits via paginação reversa."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"per_page": 1}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    if resp.status_code == 200:
        link = resp.headers.get("Link", "")
        import re
        match = re.search(r'page=(\d+)>; rel="last"', link)
        if match:
            return int(match.group(1))
        commits = resp.json()
        return len(commits)
    return 0


def get_issues_count(owner: str, repo: str, state: str = "all") -> int:
    """Conta issues (excluindo pull requests)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {"state": state, "per_page": 1, "labels": ""}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    if resp.status_code == 200:
        link = resp.headers.get("Link", "")
        import re
        match = re.search(r'page=(\d+)>; rel="last"', link)
        if match:
            return int(match.group(1))
        return len(resp.json())
    return 0


def _get(url: str, params: dict = None) -> dict | None:
    """GET com tratamento de rate-limit e erros."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 403:
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait  = max(reset - int(time.time()), 5)
                log.warning(f"Rate limit atingido. Aguardando {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            log.warning(f"HTTP {resp.status_code} para {url}")
            return None
        except requests.RequestException as e:
            log.error(f"Erro na requisição ({attempt+1}/3): {e}")
            time.sleep(3)
    return None


def repo_age_days(created_at: str) -> int:
    """Calcula a idade do repositório em dias."""
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - created).days
    except Exception:
        return 0


def extract_metrics(raw: dict) -> dict:
    """Extrai e normaliza as métricas de interesse de um item da API."""
    owner = raw["owner"]["login"]
    name  = raw["name"]

    log.info(f"  Coletando métricas detalhadas: {owner}/{name}")

    languages    = get_languages(owner, name)
    total_bytes  = sum(languages.values()) or 1
    cpp_ratio    = round(languages.get("C++", 0) / total_bytes * 100, 2)

    contributors = get_contributors_count(owner, name)
    commits      = get_commits_count(owner, name)
    open_issues  = raw.get("open_issues_count", 0)

    time.sleep(0.5)   # gentileza com a API

    return {
        # Identificação
        "repo_id":           raw["id"],
        "owner":             owner,
        "name":              name,
        "full_name":         raw["full_name"],
        "url":               raw["html_url"],

        # Métricas de popularidade
        "stars":             raw.get("stargazers_count", 0),
        "forks":             raw.get("forks_count", 0),
        "watchers":          raw.get("watchers_count", 0),

        # Métricas de atividade
        "commits_total":     commits,
        "contributors":      contributors,
        "open_issues":       open_issues,
        "has_wiki":          int(raw.get("has_wiki", False)),
        "has_projects":      int(raw.get("has_projects", False)),
        "has_downloads":     int(raw.get("has_downloads", False)),

        # Métricas de código
        "size_kb":           raw.get("size", 0),
        "cpp_ratio_pct":     cpp_ratio,
        "language_count":    len(languages),
        "primary_language":  raw.get("language", ""),

        # Métricas de manutenção / qualidade
        "license":           raw.get("license", {}).get("spdx_id", "NONE") if raw.get("license") else "NONE",
        "default_branch":    raw.get("default_branch", ""),
        "is_fork":           int(raw.get("fork", False)),
        "is_archived":       int(raw.get("archived", False)),
        "is_disabled":       int(raw.get("disabled", False)),
        "has_issues":        int(raw.get("has_issues", True)),

        # Métricas temporais
        "created_at":        raw.get("created_at", ""),
        "updated_at":        raw.get("updated_at", ""),
        "pushed_at":         raw.get("pushed_at", ""),
        "age_days":          repo_age_days(raw.get("created_at", "")),

        # Metadados
        "description":       (raw.get("description") or "").replace("\n", " ")[:200],
        "topics":            "|".join(raw.get("topics", [])),
    }


# ── Pipeline principal ───────────────────────────────────────────────────────

def main():
    log.info("=== GitHub Repo Collector — C++ Game Dev ===")
    seen_ids: set[int] = set()
    all_metrics: list[dict] = []

    for query in SEARCH_QUERIES:
        if len(all_metrics) >= TARGET_REPOS:
            break

        log.info(f"\nBuscando: '{query}'")
        repos = search_repos(query, per_page=30, max_pages=2)
        log.info(f"  {len(repos)} repositórios encontrados na busca.")

        for raw in repos:
            if len(all_metrics) >= TARGET_REPOS:
                break
            repo_id = raw["id"]
            if repo_id in seen_ids:
                continue
            seen_ids.add(repo_id)

            try:
                metrics = extract_metrics(raw)
                all_metrics.append(metrics)
                log.info(f"  [{len(all_metrics):>3}/{TARGET_REPOS}] {metrics['full_name']} "
                         f"(★{metrics['stars']} | {metrics['cpp_ratio_pct']}% C++)")
            except Exception as e:
                log.warning(f"  Erro ao processar {raw.get('full_name', '?')}: {e}")

        time.sleep(2)   # pausa entre queries

    if not all_metrics:
        log.error("Nenhum repositório coletado. Verifique seu GITHUB_TOKEN.")
        return

    df = pd.DataFrame(all_metrics)
    df.drop_duplicates(subset="repo_id", inplace=True)
    df.sort_values("stars", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    log.info(f"\n✅ Dataset salvo em '{OUTPUT_FILE}' com {len(df)} repositórios.")
    log.info(f"   Colunas: {list(df.columns)}")
    log.info(f"\nTop 5 repositórios por estrelas:\n{df[['full_name','stars','commits_total','contributors']].head()}")


if __name__ == "__main__":
    main()
