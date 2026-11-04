# Pipeline de Pré-processamento

Com base nos achados da EDA (análises univariada, bivariada, multivariada e PCA), definimos as estratégias de pré-processamento abaixo e as implementamos formalmente em um `ColumnTransformer`/`Pipeline` do scikit-learn.

## Justificativas de Pré-processamento

### Imputação de valores ausentes

- **Variável afetada:** apenas `bmi` (201 valores ausentes, ~3,9%).
- **Estratégia escolhida:** `SimpleImputer(strategy="median")`.
- **Justificativa:** a distribuição de `bmi` é assimétrica à direita e possui outliers (ver [Multivariada e Outliers](multivariada.md)); a **mediana** é menos sensível a valores extremos do que a média, evitando distorcer a imputação. Como a proporção de ausentes é pequena (~3,9%) e não há evidência de que a ausência esteja associada ao *target*, a imputação é preferível à remoção das linhas, que descartaria parte da já escassa classe minoritária de `stroke = 1`.

!!! warning "Correção feita na APS2"
    A afirmação de que a ausência de `bmi` não está associada ao *target* não se sustenta. No treino, AVC ocorre em 21,8% dos pacientes sem `bmi` registrado e em 4,1% dos demais. A imputação por mediana continua válida, mas apaga essa informação. Veja a [Mudança no Pipeline](../aps2/mudanca.md).

### Encoding de variáveis categóricas

- **Variáveis afetadas:** `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`.
- **Estratégia escolhida:** `OneHotEncoder(handle_unknown="ignore")`.
- **Justificativa:** todas essas variáveis são **nominais** (sem ordem natural entre categorias), portanto *Label Encoding* introduziria uma relação ordinal artificial (ex.: sugerir que "Private" > "Self-employed"), o que enviesaria modelos baseados em distância ou lineares. O One-Hot Encoding evita essa ordenação espúria. O parâmetro `handle_unknown="ignore"` garante robustez caso o conjunto de teste (ou dados futuros) contenha categorias não vistas no treino.

### Normalização/Padronização de variáveis numéricas

- **Variáveis afetadas:** `age`, `avg_glucose_level`, `bmi`.
- **Estratégia escolhida:** `StandardScaler` (padronização z-score: média 0, desvio padrão 1).
- **Justificativa:** as três variáveis numéricas estão em escalas muito diferentes (idade em anos, glicose em mg/dL na casa das centenas, IMC em dezenas), o que prejudicaria algoritmos sensíveis à escala (regressão logística regularizada, KNN, SVM, PCA). Como identificado no diagnóstico de outliers, há valores extremos em `avg_glucose_level` e `bmi` que são clinicamente relevantes e não devem ser removidos; por isso, documentamos `RobustScaler` (baseado em mediana e IQR) como alternativa a ser comparada na APS2 caso o `StandardScaler` se mostre excessivamente influenciado pelos valores extremos durante a modelagem.

## Separação Treino/Teste Estratificada

Utilizamos `train_test_split` com `stratify=y` para preservar a proporção original da classe minoritária (`stroke = 1`, ~4,9%) tanto no treino quanto no teste — essencial dado o forte desbalanceamento identificado na análise inicial. `random_state=42` fixo garante reprodutibilidade.

```python
X = df.drop(columns=["stroke"])
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)
```

| Conjunto | Instâncias | Proporção `stroke = 1` |
|---|---:|---:|
| Treino | 4.088 | 4,87% |
| Teste | 1.022 | 4,89% |

## ColumnTransformer + Pipeline

Construímos dois sub-pipelines (numérico e categórico) combinados por um `ColumnTransformer`, encapsulados em um `Pipeline` final. Assim, `fit` é chamado **apenas sobre `X_train`**, e `transform` é aplicado depois em `X_test` — evitando qualquer vazamento de informação do teste (estatísticas de mediana/escala/categorias vêm exclusivamente do treino).

```python
numeric_features = ["age", "avg_glucose_level", "bmi"]
categorical_features = [
    "gender", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "smoking_status",
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

preprocessing_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
```

`hypertension` e `heart_disease` já são binárias (0/1) no dataset original, mas são tratadas como categóricas nominais para manter a semântica de "sim/não" e permitir consistência caso surjam novas categorias.

Após o `fit_transform` no treino, o pipeline produz 23 features finais (3 numéricas + 20 colunas one-hot).

## Verificação: Ausência de Data Leakage

```python
X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

imputer_fitted = (
    preprocessing_pipeline
    .named_steps["preprocessor"]
    .named_transformers_["num"]
    .named_steps["imputer"]
)
assert abs(imputer_fitted.statistics_[2] - X_train["bmi"].median()) < 1e-9
```

A mediana de `bmi` aprendida pelo `SimpleImputer` (28,0) é idêntica à mediana calculada manualmente apenas em `X_train`, confirmando que **todas as estatísticas do pipeline foram ajustadas exclusivamente no conjunto de treino** — nenhuma informação do conjunto de teste vazou para o pré-processamento.

## Exportação do Pipeline

```python
import joblib
joblib.dump(preprocessing_pipeline, "artifacts/stroke_preprocessing_pipeline.joblib")
```

O `Pipeline` de pré-processamento, já ajustado no conjunto de treino, é exportado para reutilização direta na APS2 (etapa de modelagem/classificação), garantindo consistência entre as etapas e evitando reprocessamento manual.
