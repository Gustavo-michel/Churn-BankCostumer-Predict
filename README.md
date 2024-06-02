# Churn Bank Costumer Predict

![churnimg](https://github.com/Gustavo-michel/Churn-BankCostumer-Predict/assets/127684360/132fcbd7-84bc-479a-94ba-92e877efe485)

Dataset: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

## Sobre
Este projeto visa desenvolver um sistema de previsão de churn utilizando técnicas de análise exploratória de dados e machine learning. O dataset utilizado neste projeto contém informações de clientes, onde cada observação é rotulada como churn ou não churn. As variáveis presentes no dataset incluem características demográficas, comportamentais e de transações, e herarquia dos clientes.

Acessar pagina web: ___

## Análise dos Dados

### Descrição dos Dados
- Limpeza e Preparação dos Dados: Remoção de valores ausentes e duplicados*.
- Exploração de Dados: Análise estatística descritiva para entender a distribuição das variáveis e identificar possíveis padrões ou discrepâncias(outliers).

### Visualização Gráfica
- Verificação da dispersão dos dados de acordo com a classe.
- Distribuição das variáveis principais.
- Detecção dos valores dispersos no DataFrame.
- Verificação de correlação através da matriz de confusão.

![correlação_colunas](https://github.com/Gustavo-michel/Churn-BankCostumer-Predict/assets/127684360/2fd6abb8-c524-4bec-b371-274824d272b9)

### Distribuição das Variáveis
- Análise da distribuição das variáveis independentes.
- Identificação de variáveis com maior impacto no churn.

![plotboxes0](https://github.com/Gustavo-michel/Churn-BankCostumer-Predict/assets/127684360/c1d41e54-10ef-4fd3-ad85-2c42d4e1e13a)
******
![plotHist0](https://github.com/Gustavo-michel/Churn-BankCostumer-Predict/assets/127684360/6a89bd79-9482-4452-b351-f1a9f7c38de4)

### Balanceamento de Classes com SMOTE
O balanceamento de classes é uma etapa crucial em problemas de classificação, especialmente quando há um desequilíbrio significativo entre as classes, como é comum em problemas de previsão de churn. SMOTE (Synthetic Minority Over-sampling Technique) é uma técnica de oversampling que gera exemplos sintéticos da classe minoritária para igualar a distribuição das classes. Isso ajuda o modelo a aprender com mais eficácia padrões relacionados à classe minoritária, melhorando assim o desempenho da previsão.

### Seleção de Variáveis com VarianceThreshold
Nem todas as variáveis em um conjunto de dados contribuem igualmente para a previsão do resultado. Algumas variáveis podem conter pouca ou nenhuma informação relevante. VarianceThreshold é uma técnica de seleção de características que remove variáveis com baixa variância, ou seja, variáveis que têm pouca variação nos dados. Isso pode ajudar a reduzir o tempo de treinamento do modelo e melhorar a capacidade de generalização, focando apenas nas características mais informativas.

### Normalização com MinMaxScaler
A normalização é uma etapa importante no pré-processamento de dados para garantir que todas as variáveis estejam na mesma escala. O MinMaxScaler é uma técnica de normalização que transforma os dados de forma que fiquem dentro de um intervalo específico, geralmente entre 0 e 1. Isso é útil quando as variáveis têm escalas diferentes e o modelo pode interpretar erroneamente a importância relativa das características com base em suas magnitudes. A normalização ajuda a garantir que todas as características tenham um peso equivalente durante o treinamento do modelo.

### Separação de Features e Target
- Separação do dataframe em variáveis independentes (X) e variável dependente (y).

## Modelagem dos Dados
### Arquivo: model.ipynb
- Separação dos dados em conjunto de treino e teste.
- Treino e teste do modelo.
- Avaliação dos modelos utilizando pipelines e técnicas do scikit-learn
- O modelo escolhido foi LightGBMClassifier 99% de eficácia.

![Comparando modelos ROC-CURVE](https://github.com/Gustavo-michel/Churn-BankCostumer-Predict/assets/127684360/610b7f3e-9677-43d6-99d8-377030fface6)

![Matriz evaluate](https://github.com/Gustavo-michel/Churn-BankCostumer-Predict/assets/127684360/13f85fcf-f6c2-484e-9123-3eb6b49e334d)

## Acesso ao Código-Fonte
O código-fonte está disponível no repositório do GitHub. Para instalar as dependências necessárias, execute o seguinte comando:
```bash
pip install -r requirements.txt
```

No `app.py`, execute o código no servidor para iniciar a aplicação web.

## Acesso ao Modelo
O modelo treinado pode ser acessado através do site oficial do projeto. Para carregar o modelo usando a biblioteca pickle em Python, utilize o seguinte código:
```python
import pickle

# Carregar o modelo
with open('churn_detection_clf.sav', 'rb') as file:
    model = pickle.load(file)

# Com joblib
import joblib
model = joblib.load('churn_detection_clf.sav')

# Baixar arquivo pkl
with open('churn_detection_clf.pkl', 'rb') as file:
    model = pickle.load(file)

# Com joblib
import joblib
model = joblib.load('churn_detection_clf.pkl')

# Utilizar o modelo para fazer previsões
resultado = model.predict(dados)
```

## Execução do Projeto
Para executar o projeto na sua própria máquina, siga os passos abaixo:
1. Clone o repositório do GitHub.
2. Instale as dependências necessárias:
    ```bash
    pip install -r requirements.txt
    ```
3. Execute o `app.py` para iniciar a aplicação web:
    ```bash
    python src/app.py
    ```

## Contato
Telefone: +55 (11) 99434-5046  
Email: Gustavomichelads@gmail.com

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request ao repositório.

## Dashboard
Um dashboard interativo com Power BI estará disponível no projeto em breve.
