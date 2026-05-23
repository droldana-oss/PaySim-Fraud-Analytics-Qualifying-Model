# Reporte de evaluación del modelo de fraude — PaySim (Parte 2)

## Qué hice

Entrené un único `RandomForestClassifier` con datos anteriores y elegí su umbral en validación. Luego evalué el periodo futuro reservado sin modificar el modelo.

## Por qué escogí este modelo

Escogí Random Forest porque el objetivo es una clasificación binaria, las señales del EDA pueden interactuar de forma no lineal y el modelo permite ponderar la clase fraudulenta e interpretar importancias.

## Resultados en test futuro

                                threshold  precision    recall        f2  average_precision   roc_auc  verdaderos_positivos  falsos_positivos  falsos_negativos  verdaderos_negativos  monto_fraude_detectado  monto_fraude_total  monto_detectado_pct
regla                                                                                                                                                                                                                                                 
Random Forest propuesto          0.858498        1.0  0.997254  0.997802           0.999986  1.000000                  3995                 0                11                914611            6.262661e+09        6.323047e+09            99.044997
Alerta original isFlaggedFraud   0.500000        1.0  0.003245  0.004053           0.007592  0.501623                    13                 0              3993                914611            7.095204e+07        6.323047e+09             1.122118

## Conclusión

- Mi modelo detectó 3995 fraudes de 4006 (recall 99.73%).
- La alerta original detectó 13 fraudes en el mismo periodo (recall 0.32%).
- Mi modelo priorizó 99.04% del monto fraudulento; la alerta original priorizó 1.12%.

## Límite de interpretación

Este resultado es educativo y usa datos sintéticos. Incluye saldos posteriores, por lo que corresponde a revisión post-transacción y no a autorización previa en producción.

Un desempeño muy alto puede deberse a patrones simulados de PaySim y a la disponibilidad de saldos posteriores; no demuestra rendimiento equivalente en datos bancarios reales.
