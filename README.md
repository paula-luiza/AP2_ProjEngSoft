# Predição de Popularidade de Jogos Open Source em C++

Trabalho final de Machine Learning Aplicado à Mineração de Repositórios de
Código-Fonte. O objetivo é prever a popularidade (`stars`) de jogos open
source em C++ no GitHub, usando métricas de atividade, código e processo
de desenvolvimento.

---

## Pipeline (ordem de execução)

```
github_repo_collector.py          → cpp_game_repos.csv
02_limpeza_e_feature_engineering.py → cpp_game_repos_clean.csv + feature_columns.txt
03_treino_e_avaliacao_modelos.py    → resultados_*.csv + figures/
```

### 1. Setup do ambiente

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install requests pandas python-dotenv scikit-learn matplotlib seaborn
```

Criar um arquivo `.env` (salvo em UTF-8, pelo VS Code) com:
```
GITHUB_TOKEN=seu_token_aqui
```

### 2. Coleta de dados (`github_repo_collector.py`)

- Busca **150 repositórios** de jogos open source jogáveis em C++ via API do GitHub
- Inclui uma lista "pinada" de jogos famosos (DOOM, Quake, 0 A.D., Cataclysm-DDA,
  SuperTuxKart, Minetest, Endless Sky, Teeworlds, etc.) + busca por queries
- Um filtro automático (`is_likely_game`) remove engines, emuladores e
  bibliotecas que apareçam na busca
- Gera `cpp_game_repos.csv` com 37 colunas: popularidade, atividade,
  código, issues, bugs (via label "bug"), datas, licença, tópicos

### 3. Limpeza e engenharia de atributos (`02_limpeza_e_feature_engineering.py`)

Script em células (`# %%`, roda no VS Code Interactive Window). Principais
decisões — todas documentadas com comentários no código, prontas para virar
texto da seção de Metodologia:

- **Variável alvo:** `log_stars = log1p(stars)` (distribuição original é
  muito assimétrica — skew 3.83 → 0.60)
- **Data leakage removido:** `watchers` (correlação 1.00 com stars) e
  `forks` (0.90) — saem das features
- **Valores ausentes:** `bug_resolution_rate` e `issue_close_rate_pct`
  preenchidos com 0 (ausência = projeto não usa essa convenção)
- **Multicolinearidade perfeita removida:** `bugs_total` e `total_issues`
  são somas exatas de outras colunas → removidas das features
- **Outlier extremo tratado:** `CleverRaven/Cataclysm-DDA` tem até 318x os
  valores medianos em várias métricas. Resolvido aplicando `log1p` em
  variáveis de contagem (`size_kb`, `commits_total`, `contributors`,
  `releases`, `open_issues`, etc.)
- **Novas features:** `commits_per_contributor`, `bug_density`,
  `issues_per_contributor`, `releases_per_year`, flags de gênero
  (`topic_roguelike`, `topic_rpg`, `topic_strategy`, `topic_multiplayer`),
  licença codificada (one-hot, categorias raras agrupadas em "OTHER")

**Resultado:** `cpp_game_repos_clean.csv` (150 linhas, 54 colunas) +
`feature_columns.txt` (36 features usadas no modelo)

### 4. Treino e avaliação (`03_treino_e_avaliacao_modelos.py`)

Compara 3 modelos de regressão:

- **Regressão Linear (Ridge)** — com `RidgeCV` para escolher a regularização
- **Random Forest**
- **Gradient Boosting**

Avaliação em duas etapas:
1. Validação cruzada (5-fold) no treino — comparação robusta entre modelos
2. Avaliação final no conjunto de teste (20%, hold-out)

Métricas: RMSE, MAE e R² (em `log(stars)` e convertidas para "estrelas reais")

---

## Resultados atuais (validação cruzada, 5-fold)

| Modelo | RMSE (log) | Desvio | MAE (log) | R² |
|---|---|---|---|---|
| Regressão Linear (Ridge) | 1.09 | 0.11 | 0.90 | 0.62 |
| Random Forest | 1.01 | 0.05 | 0.81 | **0.67** |
| Gradient Boosting | 1.06 | 0.10 | 0.84 | 0.64 |

**Random Forest é o melhor modelo** (maior R², menor desvio entre folds).

### Features mais importantes (Random Forest / Gradient Boosting)
1. `contributors`
2. `days_since_update`
3. `age_days`
4. `open_issues`
5. `commits_per_contributor`

**Interpretação:** projetos mais antigos, com mais contribuidores e
atividade recente tendem a ser mais populares. Curiosamente, métricas de
bugs isoladamente não se correlacionam com popularidade.

---

## Arquivos gerados

```
cpp_game_repos.csv              # dataset bruto (150 repos)
cpp_game_repos_clean.csv        # dataset limpo + features
feature_columns.txt             # lista das 36 features usadas
resultados_cross_validation.csv # comparação dos 3 modelos (CV)
resultados_teste_final.csv      # avaliação no conjunto de teste
figures/
  ├── distribuicao_stars.png
  ├── correlacao_com_stars.png
  ├── previsto_vs_real.png
  ├── importancia_features.png
  └── coeficientes_regressao_linear.png
```

---

## Próximos passos

- [ ] Escrever o artigo científico (formato SBC, 6-10 páginas, Overleaf)
  - Introdução
  - Fundamentação Teórica
  - Metodologia (usar as decisões documentadas nos comentários dos scripts)
  - Resultados e Discussão (tabela + gráficos já gerados)
  - Conclusão
  - Referências
- [ ] Preparar apresentação
- [ ] (Opcional) Testar outros algoritmos/hiperparâmetros para comparação extra
