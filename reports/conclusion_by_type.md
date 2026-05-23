# Conclusión — Fraude por tipo de transacción

- Tipos con fraude:
    - TRANSFER: 4097 fraudes — monto fraudulento: 6,067,213,184.01
    - CASH_OUT: 4116 fraudes — monto fraudulento: 5,989,202,243.83

- Interpretación del patrón observado:
    - En este dataset, `TRANSFER` y `CASH_OUT` son operaciones que implican salida de fondos desde la cuenta origen y concentran todos los casos etiquetados como fraude.
    - Ambos presentan montos fraudulentos altos y con frecuencia dejan el saldo origen en cero, una señal útil para el análisis y el modelo.

- Tipos sin fraude etiquetado en PaySim (`CASH_IN`, `PAYMENT`, `DEBIT`):
    - No se observan casos `isFraud = 1` para estos tipos en los datos analizados.
    - Esta observación describe el dataset sintético y no permite concluir que esas operaciones sean libres de fraude en contextos reales.
