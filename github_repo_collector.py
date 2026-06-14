"""
GitHub Repository Collector — Jogos Open Source em C++
=======================================================
Foco exclusivo em jogos jogáveis open source escritos em C++.
Engines, emuladores e bibliotecas são excluídos.

Requisitos:
    pip install requests pandas python-dotenv

Uso:
    1. Defina GITHUB_TOKEN no arquivo .env
    2. Execute: python github_repo_collector.py
"""

import os
import re
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ── Configuração ─────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
OUTPUT_FILE  = "cpp_game_repos.csv"
TARGET_REPOS = 150

# Jogos jogáveis open source em C++ — pinados para garantir presença no dataset
# Critério: deve ser um jogo que o usuário final joga, não uma ferramenta/engine
PINNED_REPOS = [
    # FPS / Shooters
    ("id-Software",      "DOOM"),
    ("id-Software",      "Quake"),
    ("id-Software",      "Quake-III-Arena"),
    ("id-Software",      "Quake-2"),
    ("chocolate-doom",   "chocolate-doom"),
    ("fabiangreffrath",  "crispy-doom"),

    # Estratégia / Tempo Real
    ("0ad",              "0ad"),
    ("wesnoth",          "wesnoth"),
    ("spring",           "spring"),
    ("unknown-horizons", "unknown-horizons"),
    ("openage",          "openage"),

    # RPG / Aventura
    ("CleverRaven",      "Cataclysm-DDA"),
    ("gemrb",            "gemrb"),
    ("scummvm",          "scummvm"),         # engine de point-and-click com jogos próprios
    ("OpenMW",           "openmw"),          # reimplementação de Morrowind jogável

    # Corrida / Simulação
    ("supertuxkart",     "stk-code"),
    ("flightgear",       "flightgear"),

    # Plataforma / Arcade
    ("SuperTuxProject",  "SuperTux"),
    ("SuperTuxProject",  "supertux"),

    # Espaço / Exploração
    ("endless-sky",      "endless-sky"),
    ("pioneer-space",    "pioneer"),

    # Sandbox / Outros
    ("minetest",         "minetest"),
    ("the-mana-world",   "manaplus"),
    ("freecol",          "freecol"),
    ("lincity-ng",       "lincity-ng"),
    ("megaglest",        "megaglest-source"),
    ("worldforge",       "worldforge"),
    ("teeworlds",        "teeworlds"),
    ("xonotic",          "xonotic"),
    ("ioquake3",         "ioquake3"),
    ("assaultcube",      "assaultcube"),
    ("sauerbraten",      "sauerbraten"),
    ("red-eclipse",      "red-eclipse-base"),
    ("tremulous",        "tremulous"),
]

# Queries focadas em jogos jogáveis — excluem engines e emuladores explicitamente
SEARCH_QUERIES = [
    "open source game language:C++ stars:>50 topic:game",
    "open source game language:C++ stars:>30 topic:gameplay",
    "fps game language:C++ stars:>20",
    "strategy game language:C++ stars:>20",
    "rpg game language:C++ stars:>20",
    "platformer game language:C++ stars:>15",
    "roguelike language:C++ stars:>20",
    "arcade game language:C++ stars:>15",
    "puzzle game language:C++ stars:>15",
    "racing game language:C++ stars:>15",
    "shooter game language:C++ stars:>20",
    "simulation game language:C++ stars:>20",
    "adventure game language:C++ stars:>15",
    "multiplayer game language:C++ stars:>30",
    "indie game language:C++ stars:>20",
    "doom clone language:C++ stars:>10",
    "quake clone language:C++ stars:>10",
    "mmorpg language:C++ stars:>20",
    "survival game language:C++ stars:>20",
    "sandbox game language:C++ stars:>20",
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

# Palavras-chave que indicam que o repo NÃO é um jogo jogável
EXCLUDE_KEYWORDS = [
    "engine", "framework", "library", "lib", "sdk", "toolkit", "emulator",
    "emu", "renderer", "rendering", "physics", "audio", "gui", "ui",
    "template", "boilerplate", "tutorial", "demo", "example", "sample",
    "benchmark", "tool", "editor", "plugin", "wrapper", "binding",
]

def is_likely_game(raw: dict) -> bool:
    """Filtra repositórios que parecem ser engines, libs ou emuladores."""
    name = (raw.get("name") or "").lower()
    desc = (raw.get("description") or "").lower()
    topics = [t.lower() for t in raw.get("topics", [])]

    # Rejeita se nome ou descrição contém palavras de exclusão
    text = name + " " + desc
    for kw in EXCLUDE_KEYWORDS:
        if kw in text:
            return False

    # Aceita se tem tópicos que indicam jogo
    game_topics = {"game", "gameplay", "videogame", "game-development",
                   "fps", "rpg", "strategy", "platformer", "roguelike",
                   "shooter", "arcade", "puzzle", "simulation"}
    if game_topics & set(topics):
        return True

    # Aceita se a descrição menciona jogo diretamente
    game_words = ["game", "play", "player", "level", "score", "enemy",
                  "multiplayer", "singleplayer", "campaign", "mission"]
    if any(w in text for w in game_words):
        return True

    return False


# ── Funções de coleta ─────────────────────────────────────────────────────────

def _get(url: str, params: dict = None) -> dict | None:
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 403:
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait  = max(reset - int(time.time()), 5)
                log.warning(f"Rate limit. Aguardando {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            log.warning(f"HTTP {resp.status_code} — {url}")
            return None
        except requests.RequestException as e:
            log.error(f"Erro ({attempt+1}/3): {e}")
            time.sleep(3)
    return None


def _paginated_count(url: str, params: dict = None) -> int:
    p = dict(params or {})
    p["per_page"] = 1
    resp = requests.get(url, headers=HEADERS, params=p, timeout=15)
    if resp.status_code != 200:
        return 0
    link = resp.headers.get("Link", "")
    match = re.search(r'page=(\d+)>; rel="last"', link)
    if match:
        return int(match.group(1))
    data = resp.json()
    return len(data) if isinstance(data, list) else 0


def search_repos(query: str, per_page: int = 30, max_pages: int = 3) -> list[dict]:
    results = []
    for page in range(1, max_pages + 1):
        resp = _get("https://api.github.com/search/repositories", {
            "q": query, "per_page": per_page, "page": page,
            "sort": "stars", "order": "desc"
        })
        if not resp:
            break
        items = resp.get("items", [])
        results.extend(items)
        if len(items) < per_page:
            break
        time.sleep(1)
    return results


def get_bug_metrics(owner: str, repo: str) -> dict:
    base = f"https://api.github.com/repos/{owner}/{repo}/issues"
    open_bugs   = _paginated_count(base, {"state": "open",   "labels": "bug"})
    closed_bugs = _paginated_count(base, {"state": "closed", "labels": "bug"})
    total_bugs  = open_bugs + closed_bugs
    resolution  = round(closed_bugs / total_bugs * 100, 2) if total_bugs > 0 else None
    return {
        "bugs_open":           open_bugs,
        "bugs_closed":         closed_bugs,
        "bugs_total":          total_bugs,
        "bug_resolution_rate": resolution,
        "uses_bug_label":      int(total_bugs > 0),
    }


def repo_age_days(created_at: str) -> int:
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - created).days
    except Exception:
        return 0


def days_since(date_str: str) -> int:
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return -1


def extract_metrics(raw: dict) -> dict:
    owner = raw["owner"]["login"]
    name  = raw["name"]

    languages   = _get(f"https://api.github.com/repos/{owner}/{name}/languages") or {}
    total_bytes = sum(languages.values()) or 1
    cpp_ratio   = round(languages.get("C++", 0) / total_bytes * 100, 2)

    contributors = _paginated_count(
        f"https://api.github.com/repos/{owner}/{name}/contributors", {"anon": "true"})
    commits      = _paginated_count(
        f"https://api.github.com/repos/{owner}/{name}/commits")
    releases     = _paginated_count(
        f"https://api.github.com/repos/{owner}/{name}/releases")
    ci_workflows = (_get(f"https://api.github.com/repos/{owner}/{name}/actions/workflows") or {}).get("total_count", 0)
    bug_metrics  = get_bug_metrics(owner, name)

    age_days     = repo_age_days(raw.get("created_at", ""))
    open_issues  = raw.get("open_issues_count", 0)
    total_issues = _paginated_count(
        f"https://api.github.com/repos/{owner}/{name}/issues", {"state": "all"})
    closed_issues    = max(total_issues - open_issues, 0)
    issue_close_rate = round(closed_issues / total_issues * 100, 2) if total_issues > 0 else None
    commits_per_day  = round(commits / age_days, 4) if age_days > 0 else 0

    time.sleep(0.5)

    return {
        # Identificação
        "repo_id":              raw["id"],
        "owner":                owner,
        "name":                 name,
        "full_name":            raw["full_name"],
        "url":                  raw["html_url"],

        # Popularidade — variável alvo da predição
        "stars":                raw.get("stargazers_count", 0),
        "forks":                raw.get("forks_count", 0),
        "watchers":             raw.get("watchers_count", 0),

        # Métricas de código
        "size_kb":              raw.get("size", 0),
        "cpp_ratio_pct":        cpp_ratio,
        "language_count":       len(languages),

        # Métricas de atividade / processo
        "commits_total":        commits,
        "commits_per_day":      commits_per_day,
        "contributors":         contributors,
        "releases":             releases,
        "ci_workflows":         ci_workflows,
        "has_wiki":             int(raw.get("has_wiki", False)),
        "has_projects":         int(raw.get("has_projects", False)),

        # Métricas de issues
        "open_issues":          open_issues,
        "total_issues":         total_issues,
        "closed_issues":        closed_issues,
        "issue_close_rate_pct": issue_close_rate,

        # Métricas de bugs
        "bugs_open":            bug_metrics["bugs_open"],
        "bugs_closed":          bug_metrics["bugs_closed"],
        "bugs_total":           bug_metrics["bugs_total"],
        "bug_resolution_rate":  bug_metrics["bug_resolution_rate"],
        "uses_bug_label":       bug_metrics["uses_bug_label"],

        # Métricas temporais
        "age_days":             age_days,
        "days_since_push":      days_since(raw.get("pushed_at", "")),
        "days_since_update":    days_since(raw.get("updated_at", "")),

        # Qualidade / metadados
        "license":              (raw.get("license") or {}).get("spdx_id", "NONE"),
        "is_fork":              int(raw.get("fork", False)),
        "is_archived":          int(raw.get("archived", False)),
        "topics":               "|".join(raw.get("topics", [])),
        "description":          (raw.get("description") or "").replace("\n", " ")[:200],
        "created_at":           raw.get("created_at", ""),
        "pushed_at":            raw.get("pushed_at", ""),
    }


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    log.info("=== GitHub Repo Collector — Jogos Open Source C++ ===")
    seen_ids: set[int] = set()
    all_metrics: list[dict] = []

    # 1. Pinados — jogos famosos garantidos
    log.info(f"\n📌 Coletando {len(PINNED_REPOS)} jogos pinados...")
    for owner, repo in PINNED_REPOS:
        raw = _get(f"https://api.github.com/repos/{owner}/{repo}")
        if raw is None:
            log.warning(f"  Não encontrado: {owner}/{repo}")
            continue
        if raw["id"] in seen_ids:
            continue
        seen_ids.add(raw["id"])
        try:
            metrics = extract_metrics(raw)
            all_metrics.append(metrics)
            log.info(f"  [{len(all_metrics):>3}] {metrics['full_name']} ★{metrics['stars']}")
        except Exception as e:
            log.warning(f"  Erro em {owner}/{repo}: {e}")
        time.sleep(0.5)

    # 2. Busca por queries
    log.info(f"\n🔍 Buscando via queries (alvo: {TARGET_REPOS} repositórios)...")
    for query in SEARCH_QUERIES:
        if len(all_metrics) >= TARGET_REPOS:
            break
        log.info(f"\n  Query: '{query}'")
        repos = search_repos(query)
        filtered = [r for r in repos if is_likely_game(r)]
        log.info(f"  {len(repos)} encontrados → {len(filtered)} após filtro de jogos.")

        for raw in filtered:
            if len(all_metrics) >= TARGET_REPOS:
                break
            if raw["id"] in seen_ids:
                continue
            seen_ids.add(raw["id"])
            try:
                metrics = extract_metrics(raw)
                all_metrics.append(metrics)
                log.info(f"  [{len(all_metrics):>3}/{TARGET_REPOS}] {metrics['full_name']} ★{metrics['stars']}")
            except Exception as e:
                log.warning(f"  Erro em {raw.get('full_name','?')}: {e}")

        time.sleep(2)

    # 3. Salvar
    if not all_metrics:
        log.error("Nenhum repositório coletado.")
        return

    df = pd.DataFrame(all_metrics)
    df.drop_duplicates(subset="repo_id", inplace=True)
    df.sort_values("stars", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    log.info(f"\n✅ Dataset salvo: '{OUTPUT_FILE}'")
    log.info(f"   {len(df)} jogos | {len(df.columns)} colunas")
    log.info(f"\nTop 5:\n{df[['full_name','stars','commits_total','contributors','bugs_total']].head().to_string()}")


if __name__ == "__main__":
    main()
