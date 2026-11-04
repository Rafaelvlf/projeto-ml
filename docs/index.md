# Predição de AVC

## Projeto de Machine Learning

Este projeto desenvolve um pipeline completo de Machine Learning sobre o **Stroke Prediction Dataset**, para compreender os fatores demográficos e clínicos associados ao acidente vascular cerebral (AVC) e construir um modelo de classificação.

O projeto tem duas etapas:

- **[APS1: Análise Exploratória](eda/dicionario.md)**, com a EDA e o pipeline de pré-processamento.
- **[APS2: Classificação](aps2/index.md)**, com modelos, seleção de hiperparâmetros, avaliação no teste, explicabilidade e discussão crítica.

## Integrantes

- Erick Barbosa
- Rafael Lemos

## Informações do Projeto

**Disciplina:** Machine Learning  
**Instituição:** Insper  
**Entrega da APS1:** 14/09/2026  
**Entrega da APS2:** 04/11/2026  
**Dataset:** Stroke Prediction Dataset  
**Variável alvo:** `stroke`

## Resultado da APS2

O melhor modelo foi um **Gradient Boosting com indicador de `bmi` ausente**, com Average Precision de 0,240 e ROC-AUC de 0,840 no teste (o Dummy tem 0,049 e 0,500). Com o limiar adotado, detectou 33 dos 50 AVCs do teste, com 117 falsos positivos. O [resumo da APS2](aps2/index.md) traz as métricas completas e as principais limitações.

## Estrutura da APS1

A análise exploratória foi organizada nas seguintes etapas:

1. **Visão Geral e Dicionário de Dados** — inspeção inicial, descrição das variáveis, valores ausentes, inconsistências e desbalanceamento da variável alvo.
2. **Análise Univariada** — estatísticas descritivas e análise individual das principais variáveis numéricas e categóricas.
3. **Análise Bivariada** — correlações entre variáveis numéricas e relações entre variáveis categóricas e a variável alvo.
4. **Análise Multivariada e Outliers** — cruzamento de variáveis numéricas, categóricas e o target, e diagnóstico formal de outliers (IQR/Tukey).
5. **Redução de Dimensionalidade (PCA)** — padronização, variância explicada e visualização da separabilidade das classes.
6. **Pipeline de Pré-processamento** — justificativas de imputação, encoding e escalonamento, e construção do `ColumnTransformer`/`Pipeline` do scikit-learn, sem *data leakage*.
7. **Conclusão** — síntese dos principais achados e das estratégias propostas para a etapa de modelagem (APS2).

## Divisão do Trabalho na APS1

- **Membro 1 — Erick Barbosa:** Visão Geral e Dicionário de Dados, Análise Univariada e Análise Bivariada.
- **Membro 2 — Rafael Lemos:** Análise Multivariada e Diagnóstico de Outliers, PCA, Pipeline de Pré-processamento e Conclusão.

## Notebooks Reprodutíveis

Além deste site, os notebooks com todo o código executado estão no repositório do projeto:

- APS1: [`APS1_Stroke_Parte2_Membro2.ipynb`](https://github.com/Rafaelvlf/projeto-ml/blob/main/APS1_Stroke_Parte2_Membro2.ipynb)
- APS2: [`APS2_Stroke_Classificacao.ipynb`](https://github.com/Rafaelvlf/projeto-ml/blob/main/APS2_Stroke_Classificacao.ipynb)

## Dataset

O conjunto de dados utilizado é o **Stroke Prediction Dataset**, disponibilizado publicamente no Kaggle.

O dataset contém **5.110 observações e 12 atributos**, reunindo informações demográficas e clínicas dos indivíduos.

## Referências

As referências da APS2 estão em [Referências](aps2/referencias.md). As da APS1 são:

1. FEDESORIANO. *Stroke Prediction Dataset*. Kaggle, 2021. Disponível em: <https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset>
2. PEDREGOSA, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830, 2011. Disponível em: <https://scikit-learn.org/stable/>
3. JOLLIFFE, I. T.; CADIMA, J. *Principal component analysis: a review and recent developments*. Phil. Trans. R. Soc. A, 2016.
4. TUKEY, J. W. *Exploratory Data Analysis*. Addison-Wesley, 1977. (Método do intervalo interquartil — IQR — para detecção de outliers.)
