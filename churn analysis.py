import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. GENERAZIONE DATASET REALISTICO CLIENTI
# ==========================================
np.random.seed(42)
n_clienti = 1000

mesi_cliente = np.random.randint(1, 36, n_clienti)
spesa_mensile = np.random.uniform(20, 150, n_clienti)
reclami = np.random.poisson(1.2, n_clienti)
accessi_mensili = np.random.poisson(12, n_clienti)

# Logica di probabilità di churn basata su fattori realistici:
# più reclami, meno accessi e meno anzianità = maggior rischio abbandono
p_churn = (
    0.3 
    + 0.15 * reclami 
    - 0.02 * accessi_mensili 
    - 0.01 * mesi_cliente 
    + 0.002 * spesa_mensile
)
p_churn = np.clip(p_churn, 0.05, 0.95)
churn = (np.random.rand(n_clienti) < p_churn).astype(int)

df = pd.DataFrame({
    'tempo_permanenza_mesi': mesi_cliente,
    'spesa_mensile': np.round(spesa_mensile, 2),
    'reclami_assistenza': reclami,
    'accessi_ultimo_mese': accessi_mensili,
    'churn': churn
})

print("=== ANTEPRIMA DATI CLIENTI ===")
print(df.head())
print(f"\nTasso di Churn complessivo: {df['churn'].mean():.2%}\n")

# ==========================================
# 2. PREPARAZIONE TRAINING & TEST SET
# ==========================================
# X = Feature predittive, y = Variabile target da indovinare
X = df.drop('churn', axis=1)
y = df['churn']

# Dividiamo in 80% dati di addestramento e 20% dati di verifica (test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================
# 3. ADDESTRAMENTO MODELLO (RANDOM FOREST)
# ==========================================
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Previsioni sul set di test
y_pred = model.predict(X_test)

print("=== VALUTAZIONE DELLE PERFORMANCE DEL MODELLO ===")
print(f"Accuratezza complessiva: {accuracy_score(y_test, y_pred):.2%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fedele (0)', 'Abbandono (1)']))

# ==========================================
# 4. FEATURE IMPORTANCE (COSA GUIDA IL CHURN?)
# ==========================================
importanze = pd.DataFrame({
    'Feature': X.columns,
    'Importanza': model.feature_importances_
}).sort_values('Importanza', ascending=False)

print("=== FATTORI PIÙ DETERMINANTI NEL CHURN ===")
print(importanze)

# ==========================================
# 5. VISUALIZZAZIONI GRAFICHE
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.set_theme(style="whitegrid")

# Grafico 1: Matrice di Confusione
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    ax=axes[0],
    xticklabels=['Previsto Fedele', 'Previsto Abbandono'],
    yticklabels=['Reale Fedele', 'Reale Abbandono']
)
axes[0].set_title('Matrice di Confusione (Accuratezza Previsioni)', fontsize=12, pad=10)

# Grafico 2: Feature Importance Barplot
sns.barplot(
    data=importanze, 
    x='Importanza', 
    y='Feature', 
    palette='viridis', 
    ax=axes[1],
    hue='Feature',
    legend=False
)
axes[1].set_title('Feature Importance: Quali fattori guidano il Churn?', fontsize=12, pad=10)
axes[1].set_xlabel('Impatto Relativo sul Modello')

plt.tight_layout()
plt.savefig('churn_model_results.png', dpi=300)
plt.show()
