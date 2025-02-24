import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Coleta e Preparação dos Dados
df = pd.read_csv('StudentPerformanceFactors.csv') # Substitua pelo nome correto do arquivo
print(df.head())
print(df.info())

# 2. Limpeza e Pré-Processamento

# 2.1. Tratamento de Dados Faltantes (se houver)
# Neste dataset, a coluna 'Parental_Education_Level' possui um valor faltante.  Vamos tratá-lo.
# Uma opção é preencher com a moda (valor mais frequente).
df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0], inplace=True)

# 2.2.  Conversão de variáveis categóricas em numéricas
# Utilizando Label Encoding para colunas ordinais (com alguma ordem intrínseca)
# e One-Hot Encoding para colunas nominais (sem ordem).

# Instanciando o LabelEncoder
label_encoder = LabelEncoder()

# Lista de colunas para Label Encoding
label_encoding_columns = ['Attendance', 'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Peer_Influence', 'Distance_from_Home', 'Gender', 'Parental_Education_Level']

# Aplicando Label Encoding
for column in label_encoding_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Lista de colunas para One-Hot Encoding
one_hot_encoding_columns = ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities', 'School_Type']

# Aplicando One-Hot Encoding
df = pd.get_dummies(df, columns=one_hot_encoding_columns, drop_first=True) # drop_first=True para evitar multicolinearidade

#Visualizar o DataFrame após o tratamento de dados:
print("\nDataFrame após o pré-processamento:")
print(df.head())

# 3. Análise Exploratória de Dados (EDA)
# Adaptando os exemplos para o dataset

# Contagem da variável alvo ('Exam_Score')
plt.figure(figsize=(12, 6))
sns.histplot(df['Exam_Score'], kde=True)  #kde = Kernel Density Estimate
plt.title('Distribuição da Nota no Exame')
plt.xlabel('Nota no Exame')
plt.ylabel('Frequência')
plt.show()

# Boxplot da relação entre 'Hours_Studied' e 'Exam_Score'
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Hours_Studied', y='Exam_Score', data=df)
plt.title('Relação entre Horas de Estudo e Nota no Exame')
plt.xlabel('Horas de Estudo')
plt.ylabel('Nota no Exame')
plt.show()

# Mapa de calor da correlação entre as variáveis
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor da Correlação entre as Variáveis')
plt.show()

# 4. Modelagem
# Definindo as variáveis independentes (X) e a variável dependente (y)
X = df.drop('Exam_Score', axis=1)  # Feature matrix
y = df['Exam_Score']  # Target variable

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o modelo de Árvore de Decisão (Decision Tree)
model = DecisionTreeClassifier(random_state=42)  # Instanciando o modelo
# O parâmetro random_state garante que o modelo produza os mesmos resultados se executado várias vezes com os mesmos dados.

# Treinando o modelo com os dados de treinamento
model.fit(X_train, y_train)

# 5. Avaliação
# Realizando as previsões nos dados de teste
y_pred = model.predict(X_test)

# Calculando a acurácia (Accuracy) do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')

# Exibindo o relatório de classificação
print(classification_report(y_test, y_pred))

# 6. Interpretação e Visualização da Árvore

# Plotando a árvore de decisão (requer Graphviz instalado)
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, filled=True, fontsize=10)
plt.title('Árvore de Decisão')
plt.show()

# **Importância das features**
feature_importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)
print("\nImportância das Features:")
print(importance_df)

# Plote a importância das features
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Decision Tree')
plt.gca().invert_yaxis()  # Para exibir a feature mais importante no topo
plt.show()

# 7. Conclusões e Interpretações
# Análise da importância das features e estrutura da árvore
# para gerar insights sobre o desempenho dos alunos.
# Apresente suas conclusões e recomendações.