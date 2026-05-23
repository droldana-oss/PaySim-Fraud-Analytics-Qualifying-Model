#!/usr/bin/env python3
"""Crea borradores faltantes de notebooks de la Parte 1 (EDA PaySim)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


def write_nb(name, cells):
    path = NB_DIR / name
    if path.exists():
        print(f"Skipped existing {path}: se conservan la narrativa y las salidas ejecutadas.")
        return
    path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {path}")


COMMON_PATHS = """\
from pathlib import Path

# Rutas del proyecto (ejecutar notebooks desde la carpeta notebooks/)
PROJECT_ROOT = Path('..').resolve()
DATA_RAW = PROJECT_ROOT / 'data' / 'raw' / 'paysim.csv'
IMAGES_DIR = PROJECT_ROOT / 'images'
REPORTS_DIR = PROJECT_ROOT / 'reports'

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Cambia a True si tu equipo tiene poca RAM (~6.3M filas pueden usar varios GB)
USE_SAMPLE = False
SAMPLE_FRAC = 0.05  # 5% estratificado por fraude si USE_SAMPLE es True
RANDOM_STATE = 42
"""

LOAD_DATA = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['figure.dpi'] = 100

if not DATA_RAW.exists():
    raise FileNotFoundError(
        f'No encontré el dataset en {DATA_RAW}. '
        'Verifica el enlace simbólico data/raw/paysim.csv'
    )

# Tipos optimizados para ahorrar memoria
dtype_map = {
    'type': 'category',
    'nameOrig': 'string',
    'nameDest': 'string',
    'isFraud': 'int8',
    'isFlaggedFraud': 'int8',
}

print('Cargando datos... (puede tardar unos minutos)')
df = pd.read_csv(DATA_RAW, dtype=dtype_map)

if USE_SAMPLE:
    # Muestra estratificada: conserva la proporción de fraude del dataset completo
    fraud = df[df['isFraud'] == 1]
    legit = df[df['isFraud'] == 0]
    n_fraud = max(1, int(len(fraud) * SAMPLE_FRAC))
    n_legit = int(len(legit) * SAMPLE_FRAC)
    df = pd.concat([
        fraud.sample(n=min(n_fraud, len(fraud)), random_state=RANDOM_STATE),
        legit.sample(n=min(n_legit, len(legit)), random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f'Modo muestra activado: {len(df):,} filas')

print(f'Filas: {len(df):,} | Columnas: {df.shape[1]}')
df.head()
"""


def notebook_01():
    cells = [
        md(
            """# 01 — Entendimiento del problema de negocio

## ¿Qué vamos a hacer en este notebook?

Antes de tocar Python con datos pesados, vamos a **pensar como analista en un banco**.
Este cuaderno no busca calcular modelos; busca dejar claro **por qué** hacemos el análisis
y **qué preguntas** deben responder los siguientes notebooks.

> **Idea clave:** En fraude no alcanza con contar casos. Hay que saber *dónde* se concentra el riesgo
> y *cuánto dinero* está expuesto."""
        ),
        md(
            """## Contexto: PaySim

**PaySim** es un simulador de transacciones financieras basado en datos reales agregados.
Cada fila es una transacción con:

| Columna | Significado en lenguaje simple |
|---------|-------------------------------|
| `step` | Hora simulada en la que ocurre la operación (`1 step` = `1 hora`) |
| `type` | Tipo de transacción (TRANSFER, CASH_OUT, etc.) |
| `amount` | Monto movido |
| `nameOrig` / `nameDest` | Cuentas origen y destino |
| `oldbalanceOrg` / `newbalanceOrig` | Saldo antes y después en origen |
| `oldbalanceDest` / `newbalanceDest` | Saldo antes y después en destino |
| `isFraud` | **1 = fraude real** (etiqueta que nos interesa predecir después) |
| `isFlaggedFraud` | Si el sistema original marcó la operación como sospechosa |

En otras palabras, `step` me permite ordenar las transacciones en el tiempo: por ejemplo,
`step = 24` representa aproximadamente el final del primer día simulado.

Nuestro objetivo general (del README del proyecto):

> Identificar patrones de fraude para entender qué operaciones concentran mayor riesgo económico
> y qué señales podrían apoyar la priorización de alertas."""
        ),
        md(
            """## Las 3 preguntas de negocio (Parte 1)

Quédate con estas preguntas; los notebooks 02 y 04 deben poder responderlas con datos:

1. **¿Dónde se concentra el fraude?**  
   Por tipo de operación, por monto, por comportamiento de saldos o por cuentas.

2. **¿Cuánto dinero está expuesto al fraude?**  
   En banca importa el impacto económico, no solo el número de casos.

3. **¿Qué señales podrían priorizar alertas?**  
   Montos altos, tipos riesgosos, vaciamiento de cuenta, inconsistencias en saldos.

### Definición del caso de negocio (entregable de esta etapa)

> El análisis busca identificar patrones de fraude financiero para entender qué operaciones
> concentran mayor riesgo económico y qué señales podrían apoyar la priorización de alertas."""
        ),
        md(
            """## Mapa del proyecto (Parte 1 vs Parte 2)

| Fase | Notebooks | ¿Modelo? |
|------|-----------|----------|
| **Parte 1 — Análisis** | 01 (este), 02, 04 | No |
| **Parte 2 — Modelado** | 03, 05, 06 + `src/` | Sí |

### Orden recomendado de ejecución (Parte 1)

1. `01_business_understanding.ipynb` ← estás aquí  
2. `02_data_understanding.ipynb` — volumen, tasas y montos globales  
3. `04_exploratory_data_analysis.ipynb` — profundización por tipo, monto, saldos y alertas"""
        ),
        md(
            """## KPIs que construiremos después

| Bloque | KPIs principales |
|--------|------------------|
| General (nb 02) | Total transacciones, tasa de fraude, montos totales, ticket promedio |
| Por tipo (nb 04) | Fraudes por tipo, monto fraudulento, tasa por tipo |
| Por monto (nb 04) | Distribución, rangos, tasa por rango |
| Saldos (nb 04) | Vaciamiento origen, errores de balance |
| Alertas (nb 04) | Detectados vs no detectados, monto no marcado |

### Checklist antes de pasar al notebook 02

- [ ] Entiendo la diferencia entre `isFraud` (verdad) e `isFlaggedFraud` (alerta del sistema)
- [ ] Tengo claro que Parte 1 **no entrena modelos**
- [ ] Sé que el dataset está en `data/raw/paysim.csv`
- [ ] Si mi PC es lenta, activaré `USE_SAMPLE = True` en el notebook 02"""
        ),
        code(
            """# Opcional: verificar que el archivo de datos existe antes de continuar
from pathlib import Path
data_path = Path('../data/raw/paysim.csv')
if data_path.exists():
    print('✓ Dataset encontrado:', data_path.resolve())
else:
    print('✗ Falta el dataset. Crea el enlace en data/raw/paysim.csv')"""
        ),
    ]
    write_nb("01_business_understanding.ipynb", cells)


def notebook_02():
    cells = [
        md(
            """# 02 — Comprensión general del dataset

## Propósito

Conocer el **tamaño**, la **composición** y el **impacto económico inicial** del fraude.

Respondemos las preguntas del README — sección 2:

1. ¿Cuántas transacciones hay y qué porcentaje son fraudulentas?
2. ¿Cuál es el monto total transaccionado y cuánto corresponde a fraude?
3. ¿Cuál es el ticket promedio normal vs fraudulento?"""
        ),
        code(COMMON_PATHS),
        code(LOAD_DATA),
        md("## Primer vistazo: estructura y calidad básica"),
        code(
            """df.info()
print('\\n--- Valores nulos por columna ---')
print(df.isna().sum())
print('\\n--- Duplicados exactos ---')
print(df.duplicated().sum())"""
        ),
        md("## Tipos de transacción presentes"),
        code("df['type'].value_counts()"),
        md(
            """## KPIs generales de fraude

Calculamos las métricas que un equipo de riesgos pediría en una primera reunión."""
        ),
        code(
            """total_tx = len(df)
fraud_tx = int(df['isFraud'].sum())
fraud_rate = fraud_tx / total_tx

total_amount = df['amount'].sum()
fraud_amount = df.loc[df['isFraud'] == 1, 'amount'].sum()
fraud_amount_share = fraud_amount / total_amount

legit = df[df['isFraud'] == 0]
fraud = df[df['isFraud'] == 1]

avg_ticket_legit = legit['amount'].mean()
avg_ticket_fraud = fraud['amount'].mean()
median_ticket_fraud = fraud['amount'].median()

kpis = pd.Series({
    'Total transacciones': total_tx,
    'Total fraudes': fraud_tx,
    'Tasa de fraude (%)': round(fraud_rate * 100, 4),
    'Monto total transaccionado': total_amount,
    'Monto total fraudulento': fraud_amount,
    'Participación monto fraude (%)': round(fraud_amount_share * 100, 4),
    'Ticket promedio (no fraude)': avg_ticket_legit,
    'Ticket promedio (fraude)': avg_ticket_fraud,
    'Ticket mediano (fraude)': median_ticket_fraud,
})

kpis"""
        ),
        md(
            """### Cómo leer estos números

- **Tasa de fraude baja** es normal: hay muchas transacciones legítimas y pocas fraudulentas (*desbalance de clases*).
- Si el **monto fraudulento** o su participación es relevante, el riesgo económico puede ser alto aunque haya pocos casos.
- Comparar **ticket promedio** ayuda a ver si el fraude se asocia a operaciones de mayor valor."""
        ),
        md("## Visualización: distribución del fraude (conteo)"),
        code(
            """fig, ax = plt.subplots(1, 2, figsize=(12, 4))

counts = df['isFraud'].value_counts().sort_index()
ax[0].bar(['No fraude (0)', 'Fraude (1)'], counts.values, color=['#4c78a8', '#e45756'])
ax[0].set_title('Cantidad de transacciones')
ax[0].set_ylabel('Transacciones')

ax[1].pie(counts.values, labels=['No fraude', 'Fraude'], autopct='%1.4f%%', colors=['#4c78a8', '#e45756'])
ax[1].set_title('Proporción de fraude')

plt.tight_layout()
out = IMAGES_DIR / 'fraud_distribution.png'
plt.savefig(out, bbox_inches='tight')
plt.show()
print('Gráfico guardado en', out)"""
        ),
        md("## Insight esperado (rellena con tus números)"),
        code(
            """print(f'''
Insight sugerido:
Aunque las transacciones fraudulentas representan aproximadamente {kpis['Tasa de fraude (%)']}% del total,
concentran cerca de {kpis['Participación monto fraude (%)']}% del monto transaccionado.
El ticket promedio fraudulento ({kpis['Ticket promedio (fraude)']:,.2f}) vs no fraudulento
({kpis['Ticket promedio (no fraude)']:,.2f}) indica si el fraude se asocia a montos más altos.
''')"""
        ),
        md(
            """## Siguiente paso

Abre `04_exploratory_data_analysis.ipynb` para profundizar por **tipo de operación**,
**rangos de monto**, **saldos** y **efectividad de alertas** (`isFlaggedFraud`)."""
        ),
    ]
    write_nb("02_data_understanding.ipynb", cells)


def notebook_04():
    cells = [
        md(
            """# 04 — Análisis exploratorio de fraude (EDA)

## Propósito

Profundizar en **dónde** y **cómo** aparece el fraude. Este notebook cubre las secciones 3 a 6 del README:

1. Análisis por **tipo** de transacción  
2. Análisis por **monto**  
3. Análisis de **saldos** (señales tipo banca)  
4. Análisis de **alertas** existentes (`isFlaggedFraud`)

> Ejecuta primero el notebook 02 si aún no cargaste el dataset en esta sesión."""
        ),
        code(COMMON_PATHS),
        code(LOAD_DATA),
        md(
            """---
## Sección 3 — Análisis por tipo de transacción

**Preguntas:**
- ¿Qué tipo concentra más fraudes en cantidad?
- ¿Qué tipo mueve más dinero fraudulento?
- ¿Cambia la tasa de fraude según el tipo?"""
        ),
        code(
            """by_type = df.groupby('type', observed=True).agg(
    transacciones=('isFraud', 'size'),
    fraudes=('isFraud', 'sum'),
    monto_total=('amount', 'sum'),
)
monto_fraude_tipo = df.loc[df['isFraud'] == 1].groupby('type', observed=True)['amount'].sum()
by_type['monto_fraude'] = monto_fraude_tipo.fillna(0)
by_type['tasa_fraude_pct'] = (by_type['fraudes'] / by_type['transacciones'] * 100).round(4)
by_type['monto_prom_fraude'] = by_type['monto_fraude'] / by_type['fraudes'].replace(0, np.nan)

by_type.sort_values('fraudes', ascending=False)"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

order = by_type.sort_values('fraudes', ascending=False).index
sns.barplot(x=by_type.loc[order, 'fraudes'], y=order.astype(str), ax=axes[0], color='#e45756')
axes[0].set_title('Cantidad de fraudes por tipo')
axes[0].set_xlabel('Fraudes')

order2 = by_type.sort_values('monto_fraude', ascending=False).index
sns.barplot(x=by_type.loc[order2, 'monto_fraude'], y=order2.astype(str), ax=axes[1], color='#f58518')
axes[1].set_title('Monto fraudulento por tipo')
axes[1].set_xlabel('Monto')

plt.tight_layout()
out = IMAGES_DIR / 'fraud_by_type.png'
plt.savefig(out, bbox_inches='tight')
plt.show()
print('Guardado:', out)"""
        ),
        md(
            """**Insight esperado:** El fraude no se reparte igual entre tipos; algunos concentran más casos
y otros más exposición económica. Eso orienta **dónde poner controles de monitoreo**."""
        ),
        md(
            """---
## Sección 4 — Análisis por monto

Definimos rangos alineados al README (valores en la misma escala que PaySim)."""
        ),
        code(
            """bins = [0, 10_000, 100_000, 500_000, 1_000_000, np.inf]
labels = ['0-10k', '10k-100k', '100k-500k', '500k-1M', '>1M']

df['amount_range'] = pd.cut(df['amount'], bins=bins, labels=labels, right=True)

by_range = df.groupby('amount_range', observed=False).agg(
    transacciones=('isFraud', 'size'),
    fraudes=('isFraud', 'sum'),
)
monto_fraude_rango = df.loc[df['isFraud'] == 1].groupby('amount_range', observed=False)['amount'].sum()
by_range['monto_fraude'] = monto_fraude_rango.fillna(0)
by_range['tasa_fraude_pct'] = (by_range['fraudes'] / by_range['transacciones'] * 100).round(4)

by_range"""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=by_range.reset_index(), x='amount_range', y='monto_fraude', ax=ax, color='#72b7b2')
ax.set_title('Monto fraudulento por rango de transacción')
ax.set_xlabel('Rango de monto')
ax.set_ylabel('Monto fraudulento')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(IMAGES_DIR / 'fraud_by_amount_range.png', bbox_inches='tight')
plt.show()"""
        ),
        code(
            """# Comparación de distribución de montos (escala log para ver colas)
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(data=df, x='amount', hue='isFraud', log_scale=True, ax=ax, common_norm=False)
ax.set_title('Distribución de montos (escala log)')
plt.tight_layout()
plt.show()"""
        ),
        md(
            """---
## Sección 5 — Análisis de saldos

Variables derivadas (útiles después para el modelo):

- `origin_balance_diff` = saldo que "desapareció" del origen  
- `balance_error_origin` = inconsistencia: ¿old - amount = new?  
- `is_origin_balance_zero` = ¿quedó la cuenta origen en cero? (posible vaciamiento)"""
        ),
        code(
            """df['origin_balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['dest_balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
df['balance_error_origin'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
df['balance_error_dest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
df['is_origin_balance_zero'] = (df['newbalanceOrig'] == 0).astype('int8')

fraud = df['isFraud'] == 1
legit = df['isFraud'] == 0

balance_kpis = pd.Series({
    '% fraude con saldo origen en cero': round(df.loc[fraud, 'is_origin_balance_zero'].mean() * 100, 2),
    '% no fraude con saldo origen en cero': round(df.loc[legit, 'is_origin_balance_zero'].mean() * 100, 2),
    'Error balance origen (promedio |error| fraude)': df.loc[fraud, 'balance_error_origin'].abs().mean(),
    'Error balance origen (promedio |error| no fraude)': df.loc[legit, 'balance_error_origin'].abs().mean(),
    'Transacciones con |error origen| > 1': int((df['balance_error_origin'].abs() > 1).sum()),
})

balance_kpis"""
        ),
        md(
            """---
## Sección 6 — Análisis de alertas (`isFlaggedFraud`)

Matriz de casos:

| isFraud | isFlaggedFraud | Interpretación |
|---------|----------------|----------------|
| 1 | 1 | Fraude detectado por el sistema |
| 1 | 0 | **Fraude no detectado** (riesgo crítico) |
| 0 | 1 | Posible falsa alerta |
| 0 | 0 | Transacción normal |"""
        ),
        code(
            """df['alert_case'] = np.select(
    [
        (df['isFraud'] == 1) & (df['isFlaggedFraud'] == 1),
        (df['isFraud'] == 1) & (df['isFlaggedFraud'] == 0),
        (df['isFraud'] == 0) & (df['isFlaggedFraud'] == 1),
    ],
    ['Fraude detectado', 'Fraude NO detectado', 'Posible falsa alerta'],
    default='Normal',
)

alert_summary = df.groupby('alert_case').agg(
    transacciones=('amount', 'size'),
    monto_total=('amount', 'sum'),
).sort_values('transacciones', ascending=False)

fraud_total = df.loc[df['isFraud'] == 1, 'amount'].sum()
detected_amt = df.loc[(df['isFraud'] == 1) & (df['isFlaggedFraud'] == 1), 'amount'].sum()
undetected_amt = df.loc[(df['isFraud'] == 1) & (df['isFlaggedFraud'] == 0), 'amount'].sum()

detection_rate = df.loc[df['isFraud'] == 1, 'isFlaggedFraud'].mean()

alert_kpis = pd.Series({
    'Fraudes detectados (conteo)': int(((df['isFraud'] == 1) & (df['isFlaggedFraud'] == 1)).sum()),
    'Fraudes NO detectados (conteo)': int(((df['isFraud'] == 1) & (df['isFlaggedFraud'] == 0)).sum()),
    'Monto fraude detectado': detected_amt,
    'Monto fraude NO detectado': undetected_amt,
    'Tasa detección del sistema (%)': round(detection_rate * 100, 4),
})

print(alert_summary)
print('\\n--- KPIs alertas ---')
print(alert_kpis.to_string())"""
        ),
        code(
            """# Matriz de confusión simple: fraude real vs alerta (sin sklearn)
cm = pd.crosstab(df['isFraud'], df['isFlaggedFraud']).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=ax,
            xticklabels=['No alerta', 'Alerta'],
            yticklabels=['No fraude', 'Fraude'])
ax.set_xlabel('isFlaggedFraud')
ax.set_ylabel('isFraud (real)')
ax.set_title('Fraude real vs alerta del sistema')
plt.tight_layout()
out = IMAGES_DIR / 'confusion_matrix.png'
plt.savefig(out, bbox_inches='tight')
plt.show()
print('Guardado:', out)"""
        ),
        md(
            """---
## Variables candidatas para la Parte 2 (modelado)

A partir del EDA, estas columnas son buenas candidatas para ingeniería de características:

- `type` (codificar)
- `amount`, rangos de monto
- `origin_balance_diff`, `balance_error_origin`, `is_origin_balance_zero`
- Posibles ratios y flags de montos altos

**Conclusión Parte 1:**

> El análisis exploratorio permitió identificar que el fraude se concentra en ciertos tipos de transacción,
> rangos de monto y comportamientos de saldo. Estas señales pueden transformarse en variables predictivas
> para construir un modelo clasificativo de fraude."""
        ),
        code(
            """# Exportar resumen para reports/fraud_analytics_report.md
report_path = REPORTS_DIR / 'fraud_analytics_report.md'
content = '\\n'.join([
    '# Reporte de análisis de fraude — PaySim (Parte 1)',
    '',
    '> Ejecuta también `02_data_understanding.ipynb` para KPIs globales completos.',
    '',
    '## KPIs de alertas',
    alert_kpis.to_string(),
    '',
    '## KPIs de saldos',
    balance_kpis.to_string(),
    '',
    '## Resumen por tipo de transacción',
    by_type.sort_values('fraudes', ascending=False).to_string(),
    '',
    '## Resumen por rango de monto',
    by_range.to_string(),
])
report_path.write_text(content, encoding='utf-8')
print('Reporte actualizado:', report_path)"""
        ),
    ]
    write_nb("04_exploratory_data_analysis.ipynb", cells)


if __name__ == "__main__":
    NB_DIR.mkdir(parents=True, exist_ok=True)
    notebook_01()
    notebook_02()
    notebook_04()
    print("Done.")
