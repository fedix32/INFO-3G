
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# ==========================================
# 1. SIMULAZIONE DEI DATI DI A/B TEST
# ==========================================
# Scenario: 
# Gruppo A (Controllo): 2.500 visitatori, 250 conversioni (CR = 10.0%)
# Gruppo B (Variante):   2.500 visitatori, 310 conversioni (CR = 12.4%)

np.random.seed(42) # Rende riproducibile il risultato casuale

n_A = 2500
n_B = 2500

conv_A = 250
conv_B = 310

cr_A = conv_A / n_A
cr_B = conv_B / n_B

print("=== RISULTATI DESCRITTIVI DEL TEST ===")
print(f"Gruppo Controllo (A): {conv_A}/{n_A} conversioni -> CR: {cr_A:.2%}")
print(f"Gruppo Variante (B):   {conv_B}/{n_B} conversioni -> CR: {cr_B:.2%}")

# Calcolo Uplift relativo (% di miglioramento di B rispetto ad A)
uplift = ((cr_B - cr_A) / cr_A) * 100
print(f"Uplift Relativo: +{uplift:.2f}%\n")

# ==========================================
# 2. TEST STATISTICO (TWO-SAMPLE Z-TEST)
# ==========================================
# H0 (Ipotesi Nulla): I due tassi di conversione sono uguali (qualsiasi scarto è casuale)
# H1 (Ipotesi Alternativa): La variante B ha un tasso di conversione diverso/superiore ad A

successes = np.array([conv_B, conv_A])
samples = np.array([n_B, n_A])

# Calcolo di z-score e p-value
z_stat, p_value = proportions_ztest(count=successes, nobs=samples, alternative='larger')

print("=== RISULTATO DEL TEST DI IPOTESI ===")
print(f"Z-Score: {z_stat:.4f}")
print(f"p-value: {p_value:.5f}")

# Calcolo intervallo di confidenza al 95% per entrambi i gruppi
ci_low_A, ci_high_A = proportion_confint(conv_A, n_A, alpha=0.05, method='normal')
ci_low_B, ci_high_B = proportion_confint(conv_B, n_B, alpha=0.05, method='normal')

print(f"Intervallo di confidenza 95% Gruppo A: [{ci_low_A:.2%}, {ci_high_A:.2%}]")
print(f"Intervallo di confidenza 95% Gruppo B: [{ci_low_B:.2%}, {ci_high_B:.2%}]\n")

# ==========================================
# 3. VERDETTO E RACCOMANDAZIONE DI BUSINESS
# ==========================================
ALPHA = 0.05 # Soglia di confidenza 95%

print("=== DECISIONE STRATEGICA ===")
if p_value < ALPHA:
    print(" RISULTATO STATISTICAMENTE SIGNIFICATIVO (p < 0.05)")
    print("Rifiutiamo l'ipotesi nulla: la Variante B genera un miglioramento reale.")
    print("👉 Raccomandazione: Rilasciare la Variante B al 100% degli utenti per massimizzare le conversioni.")
else:
    print("❌ RISULTATO NON STATISTICAMENTE SIGNIFICATIVO (p >= 0.05)")
    print("Non possiamo escludere che la differenza sia dovuta al caso.")
    print("👉 Raccomandazione: Mantenere la versione di controllo A o prolungare il test per raccogliere più dati.")

# ==========================================
# 4. GRAFICO VISUALE COMPARATIVO
# ==========================================
df_plot = pd.DataFrame({
    'Gruppo': ['Controllo (A)', 'Variante (B)'],
    'CR': [cr_A * 100, cr_B * 100],
    'Errore': [(cr_A - ci_low_A) * 100, (cr_B - ci_low_B) * 100]
})

plt.figure(figsize=(8, 5))
sns.set_theme(style="whitegrid")

bars = plt.bar(
    df_plot['Gruppo'], 
    df_plot['CR'], 
    yerr=df_plot['Errore'], 
    capsize=6, 
    color=['#4A90E2', '#50E3C2'], 
    edgecolor='gray',
    alpha=0.9
)

plt.title('A/B Test: Confronto Conversion Rate con Intervalli di Confidenza (95%)', fontsize=12, pad=15)
plt.ylabel('Conversion Rate (%)', fontsize=11)
plt.ylim(0, max(df_plot['CR']) + 4)

# Etichette sulle barre
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('ab_test_results.png', dpi=300)
plt.show()