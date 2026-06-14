# %% [markdown]
# # Limpeza e Engenharia de Atributos — Jogos Open Source em C++
#
# Este script prepara o dataset `cpp_game_repos.csv` para a etapa de
# Machine Learning. Ele está dividido em células (`# %%`) para ser
# executado de forma interativa no VS Code (Python Interactive Window).
#
# Saída: `cpp_game_repos_clean.csv`, pronto para treino/teste dos modelos.

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Garante que a pasta de figuras existe antes de salvar qualquer gráfico
os.makedirs("figures", exist_ok=True)

df = pd.read_csv("cpp_game_repos.csv")
print(f"Dataset original: {df.shape[0]} linhas, {df.shape[1]} colunas")
df.head()


# %% [markdown]
# ## 1. Visão geral e valores ausentes
#
# Antes de qualquer tratamento, vamos entender o que temos:
# tipos de dados, estatísticas básicas e onde existem valores nulos.

# %%
print("Tipos de dados:")
print(df.dtypes)

# %%
print("\nValores ausentes por coluna:")
nulos = df.isnull().sum()
print(nulos[nulos > 0])

# %%
print("\nEstatísticas das principais métricas numéricas:")
df[["stars", "forks", "watchers", "commits_total", "contributors", "age_days"]].describe()


# %% [markdown]
# ## 2. Distribuição da variável alvo (`stars`)
#
# `stars` é o que queremos prever. Vamos visualizar sua distribuição.
# Repositórios de software costumam seguir uma distribuição "long tail":
# poucos projetos extremamente populares e muitos com poucas estrelas.
# Isso prejudica modelos de regressão, então normalmente aplicamos uma
# transformação logarítmica (`log1p`) para deixar a distribuição mais
# simétrica — o modelo aprende melhor e os erros ficam mais equilibrados.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["stars"], bins=30, ax=axes[0], color="steelblue")
axes[0].set_title(f"Distribuição de 'stars' (assimetria = {df['stars'].skew():.2f})")

sns.histplot(np.log1p(df["stars"]), bins=30, ax=axes[1], color="seagreen")
axes[1].set_title(f"Distribuição de 'log(stars)' (assimetria = {np.log1p(df['stars']).skew():.2f})")

plt.tight_layout()
plt.savefig("figures/distribuicao_stars.png", dpi=120)
plt.show()

print(
    "A transformação log1p reduz bastante a assimetria "
    f"({df['stars'].skew():.2f} -> {np.log1p(df['stars']).skew():.2f}). "
    "Por isso vamos criar a coluna 'log_stars' como variável alvo do modelo."
)


# %% [markdown]
# ## 3. Correlação entre métricas numéricas e `stars`
#
# Vamos olhar quais variáveis numéricas mais se relacionam com `stars`,
# para confirmar o problema de *data leakage* identificado anteriormente.

# %%
num_df = df.select_dtypes(include=[np.number]).drop(columns=["repo_id"])
corr_stars = num_df.corr()["stars"].sort_values(ascending=False)
print(corr_stars)

# %%
plt.figure(figsize=(6, 8))
sns.barplot(x=corr_stars.values, y=corr_stars.index, color="slateblue")
plt.title("Correlação de cada métrica com 'stars'")
plt.ylabel("")
plt.tight_layout()
plt.savefig("figures/correlacao_com_stars.png", dpi=120)
plt.show()


# %% [markdown]
# ## 4. Removendo colunas com *data leakage*
#
# Duas colunas vazam diretamente a informação que queremos prever:
#
# - **`watchers`**: tem correlação de **1.00** com `stars`. Isso ocorre
#   porque a API atual do GitHub retorna o mesmo valor para
#   `watchers_count` e `stargazers_count`. São, na prática, a mesma coluna.
# - **`forks`**: tem correlação de **~0.90**. Forks também é uma medida
#   de popularidade/adoção do projeto, não uma métrica de "atividade ou
#   código" como propõe o tema do trabalho.
#
# Mantemos essas colunas no dataset (para referência/discussão no artigo),
# mas elas **não entrarão como variáveis preditoras (features)** do modelo.

# %%
LEAKAGE_COLS = ["watchers", "forks"]
print(f"Colunas removidas das features por data leakage: {LEAKAGE_COLS}")


# %% [markdown]
# ## 5. Tratamento de valores ausentes
#
# - `bug_resolution_rate`: ausente quando o projeto não usa a label "bug"
#   (já temos a flag `uses_bug_label` indicando isso). Preenchemos com 0,
#   pois "não há bugs resolvidos registrados" é a interpretação correta —
#   não é um dado faltando por erro, é a ausência real da informação.
# - `issue_close_rate_pct`: ausente quando o projeto não tem nenhuma issue
#   (`total_issues == 0`). Mesma lógica: preenchemos com 0.

# %%
df["bug_resolution_rate"] = df["bug_resolution_rate"].fillna(0)
df["issue_close_rate_pct"] = df["issue_close_rate_pct"].fillna(0)

print("Valores ausentes após tratamento:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("(restam apenas 'topics' e 'description', que não são usadas no modelo)")


# %% [markdown]
# ## 6. Removendo colunas quase constantes
#
# `is_fork` tem apenas 1 valor diferente em 150 linhas (149 são 0).
# Uma coluna quase constante não ajuda o modelo a diferenciar nada,
# então é removida das features.

# %%
print("is_fork:", df["is_fork"].value_counts().to_dict())
NEAR_CONSTANT_COLS = ["is_fork"]


# %% [markdown]
# ## 7. Engenharia de atributos
#
# Aqui criamos novas colunas que combinam informações existentes para
# capturar melhor "intensidade de desenvolvimento" e "qualidade de processo" —
# conceitos que não aparecem em uma única métrica bruta.

# %%
# Evita divisão por zero usando np.where
df["commits_per_contributor"] = np.where(
    df["contributors"] > 0, df["commits_total"] / df["contributors"], 0
)

df["bug_density"] = np.where(
    df["commits_total"] > 0, df["bugs_total"] / df["commits_total"], 0
)

df["issues_per_contributor"] = np.where(
    df["contributors"] > 0, df["total_issues"] / df["contributors"], 0
)

df["releases_per_year"] = np.where(
    df["age_days"] > 0, df["releases"] / (df["age_days"] / 365), 0
)

print("Novas colunas criadas:")
print(df[["commits_per_contributor", "bug_density",
          "issues_per_contributor", "releases_per_year"]].describe())


# %% [markdown]
# ### 7.1 Codificação da licença (`license`)
#
# `license` é categórica com 12 valores, vários muito raros (1-2 projetos).
# Categorias raras deixam o one-hot encoding com colunas quase vazias,
# o que não ajuda o modelo. Agrupamos categorias com menos de 5 ocorrências
# em "OTHER" antes de codificar.

# %%
print("Antes do agrupamento:")
print(df["license"].value_counts())

contagem = df["license"].value_counts()
raras = contagem[contagem < 5].index
df["license_grouped"] = df["license"].replace(raras, "OTHER")

print("\nDepois do agrupamento:")
print(df["license_grouped"].value_counts())

# One-hot encoding
license_dummies = pd.get_dummies(df["license_grouped"], prefix="license", dtype=int)
df = pd.concat([df, license_dummies], axis=1)
print(f"\nColunas criadas: {list(license_dummies.columns)}")


# %% [markdown]
# ### 7.2 Flags de gênero a partir de `topics`
#
# A coluna `topics` é uma lista de tags separadas por `|`. Os gêneros mais
# frequentes no dataset são: roguelike, rpg, strategy, multiplayer.
# Criamos uma flag binária para cada um — isso pode ajudar o modelo a
# capturar se certos gêneros tendem a ter mais/menos estrelas.

# %%
GENRE_TAGS = ["roguelike", "rpg", "strategy", "multiplayer"]

df["topics"] = df["topics"].fillna("")
for tag in GENRE_TAGS:
    df[f"topic_{tag}"] = df["topics"].str.contains(tag, case=False).astype(int)

print(df[[f"topic_{t}" for t in GENRE_TAGS]].sum())


# %% [markdown]
# ### 7.3 Variável alvo: `log_stars`
#
# Conforme discutido na seção 2, o modelo será treinado para prever
# `log_stars = log(1 + stars)`. Para reportar resultados em "estrelas reais"
# depois, basta aplicar a transformação inversa: `stars = exp(log_stars) - 1`
# (a função `np.expm1` faz isso diretamente).

# %%
df["log_stars"] = np.log1p(df["stars"])


# %% [markdown]
# ## 8. Definindo o conjunto final de features
#
# Reunindo todas as decisões:
# - Remover identificadores e texto livre (não são features numéricas)
# - Remover colunas com data leakage (`watchers`, `forks`, e o próprio `stars`)
# - Remover colunas quase constantes (`is_fork`)
# - Remover colunas originais já substituídas (`license`, `license_grouped`, `topics`)

# %%
ID_COLS = [
    "repo_id", "owner", "name", "full_name", "url",
    "description", "created_at", "pushed_at",
]

COLS_TO_DROP_FROM_FEATURES = (
    ID_COLS
    + LEAKAGE_COLS
    + NEAR_CONSTANT_COLS
    + ["license", "license_grouped", "topics"]
    + ["stars"]  # variável original; usamos log_stars como alvo
)

feature_cols = [c for c in df.columns if c not in COLS_TO_DROP_FROM_FEATURES + ["log_stars"]]

print(f"Total de features para o modelo: {len(feature_cols)}")
print(feature_cols)


# %% [markdown]
# ## 9. Salvando o dataset limpo
#
# O arquivo final mantém TODAS as colunas (incluindo identificadores e
# `stars` original, para referência), mas a lista `feature_cols` acima
# indica exatamente quais colunas usar como `X` no próximo script.
#
# `log_stars` é o `y` (variável alvo).

# %%
df.to_csv("cpp_game_repos_clean.csv", index=False, encoding="utf-8")
print("Dataset limpo salvo em 'cpp_game_repos_clean.csv'")
print(f"Shape final: {df.shape}")

# Salva também a lista de features em um arquivo de texto, para reuso
with open("feature_columns.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(feature_cols))
print("Lista de features salva em 'feature_columns.txt'")
