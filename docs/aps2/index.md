# APS2: Classificação de AVC

- **Título do projeto:** Predição de AVC: classificação, avaliação e explicabilidade
- **Disciplina:** Machine Learning (Insper)
- **Grupo (dupla):** Erick Barbosa e Rafael Lemos
- **Data de entrega:** 04/11/2026
- **Dataset:** [Stroke Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Relatório público:** <https://rafaelvlf.github.io/projeto-ml/>
- **Repositório:** <https://github.com/Rafaelvlf/projeto-ml>

**Notebook com todo o código executado:** [`APS2_Stroke_Classificacao.ipynb`](https://github.com/Rafaelvlf/projeto-ml/blob/main/APS2_Stroke_Classificacao.ipynb)

## Resumo

Comparamos um Dummy, uma regressão logística regularizada e um Gradient Boosting para prever AVC em uma base com 4,9% de casos positivos, partindo do pipeline da APS1 e usando a Average Precision (AP) como métrica. O melhor modelo foi o **Gradient Boosting com um indicador de `bmi` ausente**, com AP de 0,252 na validação cruzada.

| Métrica no teste | Melhor modelo | IC 95% | Dummy |
|---|---:|---:|---:|
| Average Precision | 0,240 | 0,157 a 0,365 | 0,049 |
| ROC-AUC | 0,840 | 0,781 a 0,893 | 0,500 |
| Recall | 0,660 | 0,528 a 0,791 | 0,000 |
| Precision | 0,220 | 0,156 a 0,288 | 0,000 |
| F1 | 0,330 | 0,246 a 0,413 | 0,000 |
| Accuracy | 0,869 | 0,847 a 0,888 | 0,951 |

**Principais conclusões:**

- **O modelo ordena bem os pacientes por risco, mas a precisão é baixa.** Detectou 33 dos 50 AVCs do teste, com 117 falsos positivos, ou 4,5 alertas por AVC detectado.
- **A idade concentra o sinal.** O modelo não detectou nenhum dos 12 AVCs abaixo de 60 anos no teste.
- **O ganho do indicador não se confirmou no teste.** A validação cruzada sugeriu ganho, mas no teste a AP foi de 0,240 com o indicador e 0,251 sem ele, com intervalo de confiança que inclui ganho e perda. Para uso real, a versão sem indicador é a alternativa mais segura.
- **O recall depende bastante do limiar escolhido.** Repetir a escolha em outras divisões do treino levaria o recall no teste a valores entre 0,68 e 0,82.
- **As diferenças entre os modelos são pequenas diante do ruído** de uma base com poucos eventos. O próximo ganho tende a vir de mais casos e de variáveis clínicas melhores, e não de modelos mais complexos.

## Decisões de modelagem

| Decisão | Escolha | Justificativa |
|---|---|---|
| Métrica de seleção | Average Precision | Não depende de limiar e é informativa com classe rara |
| Validação | 5 folds estratificados, 3 repetições | Cerca de 40 casos positivos por fold exigem repetição; os mesmos folds permitem comparação pareada |
| Modelo linear | Regressão logística com `GridSearchCV` | Espaço pequeno e ajuste barato |
| Ensemble | Gradient Boosting com `RandomizedSearchCV` | Capta interações; espaço de busca grande |
| Mudança no pipeline | Indicador de `bmi` ausente | Forte associação com o AVC no treino, que a imputação da APS1 apagava |
| Melhor modelo | Maior AP média na validação cruzada | Regra definida antes de olhar o teste |
| Limiar de decisão | Máximo de F2, com predições *out-of-fold* do treino | Em uma triagem, perder um AVC custa mais do que um falso alarme |

## Como esta seção está organizada

1. [Modelos e Hiperparâmetros](modelos.md): os três modelos e a seleção de hiperparâmetros.
2. [Mudança no Pipeline e Curvas de Aprendizado](mudanca.md): o indicador de `bmi` ausente, seu impacto e o melhor modelo.
3. [Avaliação no Teste](teste.md): métricas, matriz de confusão, ROC, estabilidade do limiar e sensibilidade.
4. [Interpretação e Explicabilidade](interpretacao.md): coeficientes, importâncias e SHAP.
5. [Discussão Crítica](discussao.md): limitações, operacionalização e conclusão.
6. [Referências](referencias.md).

## Reprodutibilidade

Para reproduzir (Python 3.12 ou superior), instale as dependências com `pip install -r requirements.txt` e execute o notebook do início ao fim. A execução levou de 12 a 18 minutos em um computador de 12 núcleos, e todas as etapas aleatórias usam `random_state=42`.
