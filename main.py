import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import graphviz
import os
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=FitFailedWarning)

# 1. Coleta e Preparação dos Dados
df = pd.read_csv('StudentPerformanceFactors.csv')
print(df.head())
print(df.info())

# 2. Limpeza e Pré-Processamento
df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0], inplace=True)
label_encoder = LabelEncoder()
label_encoding_columns = ['Attendance', 'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Peer_Influence', 'Distance_from_Home', 'Gender', 'Parental_Education_Level']
for column in label_encoding_columns:
    df[column] = label_encoder.fit_transform(df[column])
one_hot_encoding_columns = ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities', 'School_Type']
df = pd.get_dummies(df, columns=one_hot_encoding_columns, drop_first=True)
print("\nDataFrame após o pré-processamento:")
print(df.head())

# 3. Análise Exploratória de Dados (EDA)

# Contagem da variável alvo ('Exam_Score')
plt.figure(figsize=(12, 6))
sns.histplot(df['Exam_Score'], kde=True)
plt.title('Distribuição da Nota no Exame')
plt.xlabel('Nota no Exame')
plt.ylabel('Frequência')
plt.show()

# Scatter plot da relação entre 'Hours_Studied' e 'Exam_Score'
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
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4.1. Escalonamento dos Dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4.2. Seleção de Features
selector = SelectKBest(score_func=f_classif, k=10)  # Seleciona as 10 melhores features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Imprimindo as features selecionadas
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]
print("Features Selecionadas:", selected_feature_names)


# 4.3. Otimização de Hiperparâmetros
param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'criterion': ['gini', 'entropy'],
    'ccp_alpha': [0.0, 0.01, 0.05, 0.1]  # Adicionando o parâmetro ccp_alpha
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

print("Melhores Parâmetros:", grid_search.best_params_)

# Usando o melhor modelo encontrado
model = grid_search.best_estimator_

# 5. Avaliação
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')
print(classification_report(y_test, y_pred, zero_division=1))


# 6. Interpretação e Visualização da Árvore
try:
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'  # Ajuste o caminho!
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=selected_feature_names, # Usando as features selecionadas
        class_names=[str(x) for x in y.unique()],
        filled=True,
        rounded=True,
        special_characters=True
    )

    graph = graphviz.Source(dot_data)
    graph.render("arvore", format="png", cleanup=True)
    graph.view()
    print("Árvore de Decisão visualizada com Graphviz (arquivo arvore.png criado).")

except Exception as e:
    print(f"Erro ao visualizar a árvore com Graphviz: {e}")
    print("Usando visualização simplificada (plot_tree com profundidade limitada).")

    model_limited = DecisionTreeClassifier(random_state=42, max_depth=3)
    model_limited.fit(X_train_selected, y_train) # Treina com as features selecionadas

    plt.figure(figsize=(20, 10))
    plot_tree(model_limited, feature_names=selected_feature_names, filled=True, fontsize=10)
    plt.title('Árvore de Decisão (Profundidade Limitada)')
    plt.show()

# **Importância das features (agora para as features selecionadas)**
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': selected_feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)
print("\nImportância das Features Selecionadas:")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Features Importantes na Árvore de Decisao (Features Selecionadas)')
plt.gca().invert_yaxis()
plt.show()

# 7. Conclusões e Interpretações
# Análise da importância das features e estrutura da árvore
# para gerar insights sobre o desempenho dos alunos.
# Apresente suas conclusões e recomendações.