# Progetto rfm
# Customer Churn Prediction with Machine Learning (Random Forest) 🤖📉

Un modello predittivo sviluppato in Python con `scikit-learn` per anticipare il rischio di abbandono dei clienti (churn) e supportare strategie proattive di retention e Customer Lifetime Value (CLV).

## 🎯 Obiettivi di Business
Acquisire nuovi clienti ha un costo sensibilmente superiore rispetto al fidelizzare quelli esistenti. Anticipare quali utenti mostrano segnali di disingaggio consente al team di marketing e customer care di intervenire con offerte personalizzate e supporto mirato prima che avvenga la disdetta.

## 🛠️ Tech Stack & Librerie
- **Linguaggio:** Python
- **Machine Learning:** `scikit-learn` (`RandomForestClassifier`, `train_test_split`, `metrics`)
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `seaborn`, `matplotlib`

## 📈 Metodologia & Pipeline
1. **Feature Engineering & Preprocessing:** Analisi di variabili comportamentali (tempo di permanenza, spesa mensile, numero di ticket/reclami aperti, frequenza di accesso).
2. **Train/Test Split:** Suddivisione del dataset con campionamento stratificato (80% training, 20% test).
3. **Model Training:** Addestramento di un classificatore Random Forest per bilanciare accuratezza e capacità di generalizzazione.
4. **Performance Evaluation:** Valutazione tramite Matrice di Confusione, Precision, Recall e F1-Score.
5. **Feature Importance:** Identificazione dei fattori più determinanti nel predire l'abbandono (es. incidenza dei reclami e frequenza di utilizzo).

## 🖼️ Visualizzazioni Output
- `churn_model_results.png`: Matrice di confusione a sinistra e grafico di importanza delle feature a destra.
