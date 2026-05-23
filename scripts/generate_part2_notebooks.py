#!/usr/bin/env python3
"""Crea los notebooks faltantes de la Parte 2 sin sobrescribir trabajo ejecutado."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source, "outputs": [], "execution_count": None}


def write_nb(name, cells):
    path = NB_DIR / name
    if path.exists():
        print(f"Skipped existing {path}: se conservan narrativa y salidas ejecutadas.")
        return
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {path}")


SETUP = """from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path('..').resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_RAW = PROJECT_ROOT / 'data' / 'raw' / 'paysim.csv'
IMAGES_DIR = PROJECT_ROOT / 'images'
REPORTS_DIR = PROJECT_ROOT / 'reports'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUTS_DIR = PROJECT_ROOT / 'data' / 'outputs'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
for folder in [IMAGES_DIR, REPORTS_DIR, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
"""

LOAD_MODELING = """import pandas as pd
from modeling_utils import (
    ALERT_COLUMN, CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET,
    TRAIN_END_STEP, VALIDATION_END_STEP, engineer_features, load_transactions,
    split_summary, temporal_split, training_sample, xy,
)

raw = load_transactions(DATA_RAW)
data = engineer_features(raw)
train, validation, test = temporal_split(data)
summary = split_summary(train, validation, test)
summary
"""


def notebook_03():
    cells = [
        md("""# 03 — Limpieza e ingeniería de variables para modelado

## ¿Qué voy a hacer en este notebook?

En la Parte 1 encontré que el fraude se concentra en `TRANSFER` y `CASH_OUT`, en montos altos y en operaciones que suelen vaciar la cuenta origen. Ahora voy a transformar esas observaciones en variables que un modelo pueda usar.

Mi objetivo todavía **no es entrenar**. Primero voy a:

1. Definir qué columnas puedo usar y cuáles debo excluir.
2. Crear variables derivadas con significado de negocio.
3. Separar entrenamiento, validación y prueba respetando el tiempo.
4. Preparar una muestra de entrenamiento que permita trabajar con el fuerte desbalance.

> Este modelo se plantea para priorizar revisión **después** de registrada la transacción, porque utilizo saldos posteriores. No lo presento como un bloqueo previo de pagos."""),
        code(SETUP),
        md("""## Carga controlada del dataset

Cargo solamente las columnas necesarias para esta fase. No necesito los identificadores de cuenta para modelar y evitarlos reduce memoria y el riesgo de que el modelo memorice cuentas concretas en vez de aprender patrones."""),
        code("""from modeling_utils import engineer_features, load_transactions

raw = load_transactions(DATA_RAW)
print(f'Transacciones cargadas: {len(raw):,}')
print(f'Fraudes reales: {int(raw["isFraud"].sum()):,}')
raw.head()"""),
        md("""## Qué variables usaré y por qué

| Decisión | Columnas | Razón |
|---|---|---|
| Incluyo información original | `type`, `amount`, saldos antes/después | El EDA mostró diferencias por tipo, monto y saldo. |
| Creo señales explicables | `log_amount`, diferencias, errores, ratio y saldo cero | Traducen comportamiento económico a números comparables. |
| Excluyo identificadores | `nameOrig`, `nameDest` | Una cuenta concreta no es una regla generalizable de fraude. |
| Excluyo la alerta previa | `isFlaggedFraud` | La usaré como comparación, no como atajo para el nuevo modelo. |
| Excluyo tiempo como predictor | `step` | Lo reservo para simular una evaluación futura sin fuga. |

`isFraud` es mi etiqueta: es lo que quiero predecir y nunca entra como variable explicativa."""),
        code("""from modeling_utils import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET

data = engineer_features(raw)
feature_table = pd.DataFrame({
    'variable_modelo': MODEL_FEATURES,
    'tipo': ['categórica'] * len(CATEGORICAL_FEATURES) + ['numérica'] * len(NUMERIC_FEATURES),
})
print('Variable objetivo:', TARGET)
feature_table"""),
        md("""## Separación temporal: entrenar con el pasado y evaluar el futuro

En PaySim, **`step` es la hora simulada en la que ocurre cada transacción**: `1 step` equivale a `1 hora`. Lo utilizo para ordenar los registros desde el pasado hacia el futuro.

No mezclo aleatoriamente transacciones antiguas y futuras. Si el objetivo es aproximarme a una alerta real, debo entrenar con operaciones anteriores y comprobar qué ocurre después.

Uso tres periodos:

- **Train:** `step <= 323`; aquí aprende el modelo.
- **Validation:** `324 <= step <= 378`; aquí escogeré el umbral de alerta en el notebook 05.
- **Test:** `step > 378`; lo guardaré sin tocar hasta la evaluación final del notebook 06.

Este corte deja aproximadamente 70% / 15% / 15% de filas por orden temporal y permite observar si el fraude cambia con el tiempo. `step` solo define estos periodos: **no se entrega al modelo como variable predictora**."""),
        code("""from modeling_utils import split_summary, temporal_split

train, validation, test = temporal_split(data)
summary = split_summary(train, validation, test)
summary"""),
        md("""## Desbalance: qué haré sin falsear la prueba

El fraude es muy escaso frente a las operaciones normales. Para que entrenar sea viable en un equipo local, conservo **todos los fraudes de train** y selecciono hasta **20 operaciones normales por cada fraude** de ese mismo periodo.

La decisión importante es esta: **no reduzco validation ni test**. Así las métricas finales siguen enfrentando la distribución que realmente aparece en esos periodos."""),
        code("""from modeling_utils import training_sample

train_sample = training_sample(train)
sampling_summary = pd.Series({
    'Filas train original': len(train),
    'Fraudes train original': int(train[TARGET].sum()),
    'Filas muestra para entrenar': len(train_sample),
    'Fraudes conservados en muestra': int(train_sample[TARGET].sum()),
    'Legítimas en muestra': int((train_sample[TARGET] == 0).sum()),
    'Tasa fraude muestra (%)': round(train_sample[TARGET].mean() * 100, 4),
})
sampling_summary"""),
        code("""summary_path = PROCESSED_DIR / 'modeling_split_summary.csv'
summary.to_csv(summary_path)
print('Resumen de particiones guardado en:', summary_path)
"""),
        md("""## Conclusión del notebook

He construido variables alineadas con los hallazgos de negocio y he definido una evaluación más honesta: aprender del pasado y medir en operaciones posteriores. El hecho de que la tasa de fraude cambie entre periodos será relevante al interpretar el modelo; un buen resultado en entrenamiento no garantiza el mismo comportamiento futuro.

El siguiente paso es abrir `05_modeling.ipynb`, donde justificaré un único clasificador, lo entrenaré con la muestra de `train` y seleccionaré el umbral usando únicamente `validation`."""),
    ]
    write_nb('03_data_cleaning_feature_engineering.ipynb', cells)


def notebook_05():
    cells = [
        md("""# 05 — Selección y entrenamiento del modelo clasificativo

## ¿Qué voy a hacer en este notebook?

Ya preparé variables y periodos sin fuga temporal. En PaySim, `step` representa la **hora simulada** de cada transacción (`1 step` = `1 hora`); lo usé para separar pasado y futuro, pero no entra como variable del modelo.

Ahora voy a escoger y entrenar **un solo modelo** para asignar una probabilidad de fraude a cada transacción.

Mi decisión es entrenar un **Random Forest clasificativo**. La etiqueta `isFraud` tiene dos valores (`0` o `1`), por eso el problema es de clasificación y no de regresión."""),
        code(SETUP),
        code(LOAD_MODELING),
        md("""## ¿Por qué escojo Random Forest?

Antes de entrenar, comparo conceptualmente las alternativas razonables:

| Modelo posible | Qué aportaría | Decisión en este proyecto |
|---|---|---|
| Regresión logística | Explicación lineal y rápida | No la entreno: puede no capturar combinaciones no lineales de monto, tipo y saldos. |
| Árbol único | Reglas fáciles de leer | No lo entreno: puede ser inestable y sobreajustar. |
| Gradient boosting | Suele rendir bien en tablas | No lo entreno: exige una búsqueda y explicación más complejas que el alcance actual. |
| **Random Forest** | Combina árboles, captura interacciones y admite pesos por clase | **Lo entreno como único modelo de esta entrega.** |

Esta elección no significa que Random Forest sea siempre el mejor modelo. Es una decisión razonada para mi primer modelo: puedo representar patrones no lineales del EDA, darle más importancia al fraude escaso y luego revisar qué variables influyen más."""),
        md("""## Preparación para entrenar

Uso únicamente la muestra de `train`: todos los fraudes históricos disponibles en ese periodo y una muestra reproducible de operaciones legítimas. `validation` queda completa y solo se utiliza después para definir el umbral de alerta."""),
        code("""from modeling_utils import build_random_forest_pipeline

train_sample = training_sample(train)
X_train, y_train = xy(train_sample)
X_validation, y_validation = xy(validation)
print(f'Entrenamiento usado por el modelo: {len(X_train):,} filas | fraudes: {int(y_train.sum()):,}')
print(f'Validación sin muestrear: {len(X_validation):,} filas | fraudes: {int(y_validation.sum()):,}')

model = build_random_forest_pipeline()
model"""),
        md("""## Entrenamiento del único modelo

El parámetro `class_weight='balanced_subsample'` hace que cada árbol preste más atención a la clase fraudulenta, que es minoritaria. Limito profundidad y tamaño mínimo de hoja para reducir sobreajuste, y fijo `random_state` para poder reproducir el resultado."""),
        code("""model.fit(X_train, y_train)
validation_probability = model.predict_proba(X_validation)[:, 1]
print('Modelo Random Forest entrenado. Probabilidades generadas solo para validation.')"""),
        md("""## Elección del umbral con validación

Una probabilidad necesita un punto de corte para convertirse en alerta. No uso automáticamente `0.50`, porque en fraude la clase positiva es muy rara y perder un fraude puede ser más costoso que revisar una alerta adicional.

Selecciono el umbral que maximiza **F2** en `validation`: esta métrica da más peso al *recall* (capturar fraudes) sin olvidar la precisión. El conjunto `test` sigue sin participar en ninguna decisión."""),
        code("""from modeling_utils import metric_summary, select_f2_threshold

threshold, validation_curve = select_f2_threshold(y_validation, validation_probability)
validation_metrics = metric_summary(validation, validation_probability, threshold, 'Random Forest - validation')
print(f'Umbral seleccionado con validation: {threshold:.6f}')
validation_metrics[['precision', 'recall', 'f2', 'average_precision', 'verdaderos_positivos', 'falsos_positivos', 'falsos_negativos']]"""),
        code("""import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(validation_curve['recall'], validation_curve['precision'], color='#e45756')
selected = validation_curve.loc[validation_curve['f2'].idxmax()]
ax.scatter(selected['recall'], selected['precision'], color='#1f77b4', s=70, label=f'Umbral F2 = {threshold:.4f}')
ax.set_title('Precisión-Recall en validación')
ax.set_xlabel('Recall (fraudes capturados)')
ax.set_ylabel('Precisión (alertas correctas)')
ax.legend()
plt.tight_layout()
validation_plot = IMAGES_DIR / 'model_validation_precision_recall.png'
plt.savefig(validation_plot, bbox_inches='tight')
plt.show()
print('Gráfico guardado:', validation_plot)"""),
        md("""## Interpretación de variables

Random Forest no produce una fórmula lineal, pero sí permite observar qué variables utilizó con más frecuencia para separar casos. Esto no demuestra causalidad; me ayuda a comprobar si el modelo está apoyándose en señales coherentes con el análisis exploratorio."""),
        code("""import pandas as pd

feature_names = model.named_steps['preprocessing'].get_feature_names_out()
importance = pd.Series(model.named_steps['model'].feature_importances_, index=feature_names).sort_values(ascending=False)
top_importance = importance.head(12)
top_importance"""),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
top_importance.sort_values().plot.barh(ax=ax, color='#4c78a8')
ax.set_title('Variables con mayor importancia en Random Forest')
ax.set_xlabel('Importancia del modelo')
plt.tight_layout()
importance_plot = IMAGES_DIR / 'model_feature_importance.png'
plt.savefig(importance_plot, bbox_inches='tight')
plt.show()
print('Gráfico guardado:', importance_plot)"""),
        md("""## Guardar el artefacto para evaluar una sola vez

Guardo el pipeline completo y el umbral escogido. Así el notebook 06 no reentrena ni cambia decisiones al ver el periodo de prueba: únicamente aplica lo que ya quedó definido."""),
        code("""import json
import joblib

bundle = {
    'model': model,
    'threshold': threshold,
    'features': MODEL_FEATURES,
    'train_end_step': TRAIN_END_STEP,
    'validation_end_step': VALIDATION_END_STEP,
    'validation_metrics': validation_metrics.to_dict(),
}
model_path = MODELS_DIR / 'random_forest_fraud.joblib'
joblib.dump(bundle, model_path)
metadata = {
    'model': 'RandomForestClassifier',
    'threshold_selected_on_validation': threshold,
    'train_rows_sampled': len(train_sample),
    'validation_rows_full': len(validation),
    'feature_count_before_encoding': len(MODEL_FEATURES),
}
(OUTPUTS_DIR / 'model_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('Modelo guardado en:', model_path)
print('Metadatos guardados en:', OUTPUTS_DIR / 'model_metadata.json')"""),
        code("""print(
    f'Conclusión de entrenamiento: seleccioné Random Forest y un umbral de {threshold:.6f} usando únicamente validation. '
    f'En ese periodo el recall fue {validation_metrics["recall"]:.2%} y la precisión {validation_metrics["precision"]:.2%}. '
    'Todavía no concluyo desempeño final: esa comprobación corresponde al periodo test en el notebook 06. '
    'Si las métricas son muy altas, recordaré que PaySim es sintético y que uso saldos posteriores.'
)"""),
        md("""## Siguiente paso

Abriré `06_model_evaluation_interpretation.ipynb` para evaluar el modelo ya congelado sobre transacciones futuras y compararlo con la alerta original `isFlaggedFraud`.

Si la validación muestra métricas casi perfectas, las interpretaré con cautela: el dataset es sintético y las señales de saldo posteriores pueden simplificar el patrón."""),
    ]
    write_nb('05_modeling.ipynb', cells)


def notebook_06():
    cells = [
        md("""# 06 — Evaluación e interpretación final del modelo

## ¿Qué voy a hacer en este notebook?

Voy a evaluar el Random Forest ya entrenado sobre el periodo `test`, que representa transacciones posteriores a las usadas para aprender y seleccionar el umbral. La referencia temporal es `step`: en PaySim cada `step` corresponde a **una hora simulada**.

Aquí `step` solo me permite comprobar el modelo en horas posteriores; no fue usado como señal para predecir fraude. Este es el momento en que puedo responder si mi propuesta mejora la priorización frente a `isFlaggedFraud`.

No volveré a entrenar ni ajustar el umbral mirando los resultados de prueba."""),
        code(SETUP),
        code(LOAD_MODELING),
        md("""## Recuperar la decisión tomada en entrenamiento

Cargo el modelo y el umbral guardados por el notebook 05. La prueba temporal permanece completa: no quito operaciones legítimas, porque necesito medir el costo real de las alertas falsas."""),
        code("""import joblib
from modeling_utils import metric_summary, original_alert_summary

model_path = MODELS_DIR / 'random_forest_fraud.joblib'
if not model_path.exists():
    raise FileNotFoundError('Primero ejecuta 05_modeling.ipynb para generar el modelo.')
bundle = joblib.load(model_path)
model = bundle['model']
threshold = bundle['threshold']
X_test, y_test = xy(test)
test_probability = model.predict_proba(X_test)[:, 1]
print(f'Test futuro: {len(test):,} transacciones | {int(y_test.sum()):,} fraudes')
print(f'Umbral congelado desde validation: {threshold:.6f}')"""),
        md("""## Métricas que sí importan en fraude

No tomaré `accuracy` como conclusión principal: con tantas operaciones normales, un sistema que no detecte fraudes podría parecer acertado. Interpreto:

- **Recall:** de todos los fraudes, cuántos logro marcar.
- **Precisión:** de todas mis alertas, cuántas eran fraude.
- **F2:** balance que da más importancia a capturar fraudes.
- **Average Precision:** calidad del ranking de probabilidades ante el desbalance.
- **Monto detectado:** dinero fraudulento que habría quedado priorizado para revisión."""),
        code("""model_metrics = metric_summary(test, test_probability, threshold, 'Random Forest propuesto')
original_metrics = original_alert_summary(test)
comparison = pd.DataFrame([model_metrics, original_metrics]).set_index('regla')
comparison[['precision', 'recall', 'f2', 'average_precision', 'verdaderos_positivos', 'falsos_positivos', 'falsos_negativos', 'monto_detectado_pct']]"""),
        md("""## Matriz de confusión y curva precisión-recall

La matriz permite ver explícitamente fraudes detectados y omitidos. La curva muestra cómo cambia el compromiso entre capturar más fraude y generar más revisiones al mover el umbral."""),
        code("""import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve

prediction = (test_probability >= threshold).astype('int8')
cm = confusion_matrix(y_test, prediction, labels=[0, 1])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=axes[0], xticklabels=['No alerta', 'Alerta'], yticklabels=['No fraude', 'Fraude'])
axes[0].set_title('Random Forest en test futuro')
axes[0].set_xlabel('Predicción')
axes[0].set_ylabel('Real')
precision, recall, _ = precision_recall_curve(y_test, test_probability)
axes[1].plot(recall, precision, color='#e45756')
axes[1].set_title('Precisión-Recall en test futuro')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precisión')
plt.tight_layout()
evaluation_plot = IMAGES_DIR / 'model_test_evaluation.png'
plt.savefig(evaluation_plot, bbox_inches='tight')
plt.show()
print('Gráfico guardado:', evaluation_plot)"""),
        md("""## Impacto económico y comparación con la alerta existente

Además del conteo, comparo cuánto monto fraudulento logra marcar cada regla. Esta lectura conecta el modelo con la pregunta de negocio original: cuánto dinero riesgoso podría priorizarse."""),
        code("""business_comparison = comparison[[
    'verdaderos_positivos', 'falsos_negativos', 'falsos_positivos',
    'monto_fraude_detectado', 'monto_fraude_total', 'monto_detectado_pct'
]].copy()
business_comparison"""),
        code("""fig, ax = plt.subplots(figsize=(9, 4))
business_comparison['monto_detectado_pct'].plot.bar(ax=ax, color=['#e45756', '#4c78a8'])
ax.set_title('Porcentaje del monto fraudulento marcado en test')
ax.set_ylabel('Monto detectado (%)')
ax.set_xlabel('Regla de alerta')
plt.xticks(rotation=0)
plt.tight_layout()
amount_plot = IMAGES_DIR / 'model_detected_amount_comparison.png'
plt.savefig(amount_plot, bbox_inches='tight')
plt.show()
print('Gráfico guardado:', amount_plot)"""),
        md("""## Interpretación responsable

Este resultado mide un ejercicio reproducible sobre PaySim, un dataset sintético. Como uso saldos posteriores a la operación, la conclusión aplica a priorizar revisión posterior, no a impedir la transacción antes de que ocurra. Un uso real requeriría datos reales, calibración, costos de falsas alertas, monitoreo de deriva y validación regulatoria.

**Advertencia sobre métricas muy altas:** PaySim puede contener patrones simulados más separables que los de un banco real. Por eso, un recall cercano al 100% aquí no demuestra que el mismo modelo funcionaría así en producción."""),
        code("""report_path = REPORTS_DIR / 'model_evaluation_report.md'
model_recall = model_metrics['recall'] * 100
alert_recall = original_metrics['recall'] * 100
model_amount = model_metrics['monto_detectado_pct']
alert_amount = original_metrics['monto_detectado_pct']
content = '\\n'.join([
    '# Reporte de evaluación del modelo de fraude — PaySim (Parte 2)',
    '',
    '## Qué hice',
    '',
    'Entrené un único `RandomForestClassifier` con datos anteriores y elegí su umbral en validación. Luego evalué el periodo futuro reservado sin modificar el modelo.',
    '',
    '## Por qué escogí este modelo',
    '',
    'Escogí Random Forest porque el objetivo es una clasificación binaria, las señales del EDA pueden interactuar de forma no lineal y el modelo permite ponderar la clase fraudulenta e interpretar importancias.',
    '',
    '## Resultados en test futuro',
    '',
    comparison.to_string(),
    '',
    '## Conclusión',
    '',
    f'- Mi modelo detectó {model_metrics["verdaderos_positivos"]:.0f} fraudes de {int(test[TARGET].sum())} (recall {model_recall:.2f}%).',
    f'- La alerta original detectó {original_metrics["verdaderos_positivos"]:.0f} fraudes en el mismo periodo (recall {alert_recall:.2f}%).',
    f'- Mi modelo priorizó {model_amount:.2f}% del monto fraudulento; la alerta original priorizó {alert_amount:.2f}%.',
    '',
    '## Límite de interpretación',
    '',
    'Este resultado es educativo y usa datos sintéticos. Incluye saldos posteriores, por lo que corresponde a revisión post-transacción y no a autorización previa en producción.',
    '',
    'Un desempeño muy alto puede deberse a patrones simulados de PaySim y a la disponibilidad de saldos posteriores; no demuestra rendimiento equivalente en datos bancarios reales.',
])
report_path.write_text(content + '\n', encoding='utf-8')
print('Reporte guardado:', report_path)
print(
    f'Conclusión final: en el periodo futuro, mi Random Forest alcanzó recall de {model_recall:.2f}% '
    f'y priorizó {model_amount:.2f}% del monto fraudulento, frente a {alert_recall:.2f}% de recall '
    f'y {alert_amount:.2f}% del monto con la alerta original.'
)"""),
        md("""## Cierre del proyecto

Con este notebook cierro la Parte 2: pasé de entender el fraude a construir y evaluar una alternativa de priorización. Mis conclusiones deben leerse junto con las limitaciones metodológicas documentadas y no como autorización para desplegar el modelo en un banco real."""),
    ]
    write_nb('06_model_evaluation_interpretation.ipynb', cells)


if __name__ == '__main__':
    NB_DIR.mkdir(parents=True, exist_ok=True)
    notebook_03()
    notebook_05()
    notebook_06()
    print('Done.')
