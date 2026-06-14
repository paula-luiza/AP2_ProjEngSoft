# %% [markdown]
# # Treino e Avaliação de Modelos — Predição de Popularidade (stars)
#
# Pré-requisito: rodar antes o script `02_limpeza_e_feature_engineering.py`,
# que gera os arquivos `cpp_game_repos_clean.csv` e `feature_columns.txt`.
#
# Modelos comparados:
# - **Regressão Linear** (baseline)
# - **Random Forest**
# - **Gradient Boosting**
#
# Métricas: RMSE, MAE e R² — calculadas tanto em escala `log(stars)`
# (escala usada no treino) quanto convertidas para "estrelas reais"
# (mais fácil de interpretar e de explicar no artigo).

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42  # fixa a aleatoriedade -> resultados reprodutíveis

# Garante que a pasta de figuras existe antes de salvar qualquer gráfico
os.makedirs("figures", exist_ok=True)


# %% [markdown]
# ## 1. Carregando dados limpos e definindo X e y

# %%
df = pd.read_csv("cpp_game_repos_clean.csv")

with open("feature_columns.txt", encoding="utf-8") as f:
    feature_cols = [line.strip() for line in f if line.strip()]

X = df[feature_cols]
y = df["log_stars"]

print(f"X: {X.shape}  |  y: {y.shape}")
print(f"Total de features: {len(feature_cols)}")


# %% [markdown]
# ## 2. Divisão treino/teste
#
# 80% treino / 20% teste. `random_state` fixo garante que, se você rodar
# o script de novo, a divisão será exatamente a mesma (reprodutibilidade —
# importante para o artigo).

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"Treino: {X_train.shape[0]} repositórios")
print(f"Teste:  {X_test.shape[0]} repositórios")


# %% [markdown]
# ## 3. Definindo os modelos
#
# **Regressão Linear (Ridge):** usamos `Ridge` em vez de `LinearRegression`
# pura. Com 38 features para apenas 120 exemplos de treino — e algumas
# features correlacionadas entre si — a Regressão Linear comum (OLS) fica
# numericamente instável e pode gerar coeficientes e previsões absurdas.
# `Ridge` adiciona um termo de regularização (penaliza coeficientes muito
# grandes), o que resolve esse problema mantendo o modelo linear e
# interpretável. `RidgeCV` escolhe automaticamente, via validação cruzada,
# a força ideal dessa regularização (`alpha`).
#
# Como é sensível à escala das variáveis, usamos um `Pipeline` com
# `StandardScaler` antes do Ridge.
#
# Random Forest e Gradient Boosting são baseados em árvores de decisão e
# não são afetados pela escala das variáveis, então não precisam de scaler.
#
# Os hiperparâmetros abaixo são conservadores (poucas árvores, profundidade
# limitada) porque temos poucos dados (150 linhas) — modelos muito complexos
# tendem a *overfitar* (memorizar o treino em vez de aprender padrões gerais).

# %%
models = {
    "Regressão Linear (Ridge)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-2, 3, 20))),
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=6, random_state=RANDOM_STATE
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
    ),
}


# %% [markdown]
# ## 4. Comparação com validação cruzada (k=5)
#
# Antes de olhar o conjunto de teste, comparamos os 3 modelos usando
# 5-fold cross-validation **dentro do conjunto de treino**. Isso dá uma
# estimativa mais estável de qual modelo tende a performar melhor,
# sem "gastar" o conjunto de teste nessa decisão.

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_results = []
for name, model in models.items():
    rmse_scores = -cross_val_score(model, X_train, y_train, cv=kf,
                                    scoring="neg_root_mean_squared_error")
    mae_scores  = -cross_val_score(model, X_train, y_train, cv=kf,
                                    scoring="neg_mean_absolute_error")
    r2_scores   =  cross_val_score(model, X_train, y_train, cv=kf,
                                    scoring="r2")

    cv_results.append({
        "Modelo": name,
        "RMSE (log) médio": rmse_scores.mean(),
        "RMSE (log) desvio": rmse_scores.std(),
        "MAE (log) médio": mae_scores.mean(),
        "R² médio": r2_scores.mean(),
    })

cv_df = pd.DataFrame(cv_results)
print("Comparação via validação cruzada (5-fold, no conjunto de treino):")
cv_df


# %% [markdown]
# ## 5. Treino final e avaliação no conjunto de teste
#
# Agora treinamos cada modelo com TODO o conjunto de treino e avaliamos
# uma única vez no conjunto de teste (que nenhum modelo viu até agora).
#
# Reportamos métricas em duas escalas:
# - **log(stars)**: escala usada no treino
# - **stars reais**: aplicando `expm1()` (inverso do `log1p`), para que o
#   erro seja interpretável em "número de estrelas"

# %%
test_results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    predictions[name] = y_pred_log

    # Métricas em escala log
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    mae_log  = mean_absolute_error(y_test, y_pred_log)
    r2_log   = r2_score(y_test, y_pred_log)

    # Convertendo para escala real de estrelas
    y_test_stars = np.expm1(y_test)
    y_pred_stars = np.expm1(y_pred_log)
    rmse_stars = np.sqrt(mean_squared_error(y_test_stars, y_pred_stars))
    mae_stars  = mean_absolute_error(y_test_stars, y_pred_stars)

    test_results.append({
        "Modelo": name,
        "RMSE (log)": rmse_log,
        "MAE (log)": mae_log,
        "R²": r2_log,
        "RMSE (estrelas)": rmse_stars,
        "MAE (estrelas)": mae_stars,
    })

test_df = pd.DataFrame(test_results)
print("Avaliação final no conjunto de TESTE:")
test_df


# %% [markdown]
# ## 6. Visualização: previsto vs. real
#
# Para cada modelo, um gráfico de dispersão entre o valor real de `stars`
# e o valor previsto. A linha diagonal representa a previsão perfeita —
# quanto mais perto da linha, melhor o modelo.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

y_test_stars = np.expm1(y_test)

for ax, (name, y_pred_log) in zip(axes, predictions.items()):
    y_pred_stars = np.expm1(y_pred_log)

    ax.scatter(y_test_stars, y_pred_stars, alpha=0.6, color="steelblue")
    max_val = max(y_test_stars.max(), y_pred_stars.max())
    ax.plot([0, max_val], [0, max_val], "r--", label="Previsão perfeita")

    ax.set_xlabel("Estrelas reais")
    ax.set_ylabel("Estrelas previstas")
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.savefig("figures/previsto_vs_real.png", dpi=120)
plt.show()


# %% [markdown]
# ## 7. Importância das features (Random Forest e Gradient Boosting)
#
# Modelos baseados em árvores permitem extrair quais variáveis tiveram
# mais peso nas decisões — útil para a seção de Resultados e Discussão
# do artigo (ex: "commits_total foi a feature mais importante...").

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, name in zip(axes, ["Random Forest", "Gradient Boosting"]):
    model = models[name]
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True).tail(10)

    importances.plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title(f"Top 10 features — {name}")
    ax.set_xlabel("Importância")

plt.tight_layout()
plt.savefig("figures/importancia_features.png", dpi=120)
plt.show()


# %% [markdown]
# ## 8. Coeficientes da Regressão Linear (Ridge)
#
# Diferente das árvores, a Regressão Linear tem coeficientes
# interpretáveis diretamente: o sinal indica se a relação é positiva
# ou negativa, e a magnitude (com dados padronizados) indica a força
# da relação. Com Ridge, os coeficientes ficam "encolhidos" (mais
# próximos de zero) em comparação ao OLS puro — isso é esperado e é
# o que garante a estabilidade do modelo.

# %%
linear_model = models["Regressão Linear (Ridge)"].named_steps["model"]
print(f"Alpha escolhido pelo RidgeCV: {linear_model.alpha_:.4f}")

coefs = pd.Series(linear_model.coef_, index=feature_cols).sort_values()

fig, ax = plt.subplots(figsize=(8, 10))
coefs.plot(kind="barh", ax=ax, color="seagreen")
ax.set_title("Coeficientes da Regressão Linear / Ridge (dados padronizados)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("figures/coeficientes_regressao_linear.png", dpi=120)
plt.show()


# %% [markdown]
# ## 9. Salvando resultados
#
# As tabelas de comparação (`cv_df` e `test_df`) são salvas em CSV — você
# pode abrir no Excel/Google Sheets e formatar como tabela para o artigo,
# ou usar `.to_latex()` se o template do Overleaf aceitar.

# %%
cv_df.to_csv("resultados_cross_validation.csv", index=False)
test_df.to_csv("resultados_teste_final.csv", index=False)

print("Resultados salvos:")
print(" - resultados_cross_validation.csv")
print(" - resultados_teste_final.csv")
print("\nGráficos salvos em 'figures/':")
print(" - previsto_vs_real.png")
print(" - importancia_features.png")
print(" - coeficientes_regressao_linear.png")

print("\n=== RESUMO FINAL (conjunto de teste) ===")
print(test_df.to_string(index=False))
