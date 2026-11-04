# Predição de AVC

Projeto da disciplina de Machine Learning (Insper) sobre o [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

**Grupo:** Erick Barbosa e Rafael Lemos

**Relatório completo:** <https://rafaelvlf.github.io/projeto-ml/>

## Etapas

| Etapa | Conteúdo | Notebook |
|---|---|---|
| APS1 | Análise exploratória e pipeline de pré-processamento | [`APS1_Stroke_Parte2_Membro2.ipynb`](APS1_Stroke_Parte2_Membro2.ipynb) |
| APS2 | Classificação, avaliação no teste, SHAP e discussão crítica | [`APS2_Stroke_Classificacao.ipynb`](APS2_Stroke_Classificacao.ipynb) |

## Como reproduzir a APS2

Requer Python 3.12 ou superior.

```bash
pip install -r requirements.txt
```

Depois, execute `APS2_Stroke_Classificacao.ipynb` do início ao fim, a partir da raiz do repositório. A execução levou de 12 a 18 minutos em um computador de 12 núcleos. As etapas aleatórias usam `random_state=42`, e as figuras são gravadas em `docs/assets/images/aps2/`.

## Como visualizar o relatório localmente

```bash
mkdocs serve
```

Cada envio para a branch `main` publica o site no GitHub Pages por meio do GitHub Actions.
