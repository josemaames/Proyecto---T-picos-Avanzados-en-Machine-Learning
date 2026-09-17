# Traductor Inglés → Español (Transformer desde cero)

Proyecto para el curso de Tópicos Avanzados en Machine Learning. Implementa un modelo
**Transformer encoder-decoder desde cero** (sin modelos ni pesos pre-entrenados) para traducir
inglés → español, entrenado sobre el corpus Tatoeba (Anki).

## Archivos

- **`mt_en_es_transformer.ipynb`** — notebook principal. Descarga el dataset, limpia el texto,
  construye el vocabulario, define la arquitectura Transformer (atención multi-cabeza,
  positional encoding, encoder/decoder — todo implementado a mano en PyTorch), entrena el
  modelo, evalúa en un conjunto de test de 100+ ejemplos con BLEU y muestra casos buenos y
  casos donde falla. Al final guarda:
  - `transformer_en_es.pth` (pesos del modelo)
  - `vocab_en.json`, `vocab_es.json`, `model_config.json` (vocabularios y configuración)
  - `resultados_test.csv` (traducciones y BLEU por ejemplo del conjunto de test)
- **`mt_en_es_inferencia.ipynb`** — notebook liviano que carga el modelo ya entrenado (los 4
  archivos anteriores) y traduce texto nuevo, sin reentrenar.

## Cómo correrlo en Google Colab

1. Sube `mt_en_es_transformer.ipynb` a [colab.research.google.com](https://colab.research.google.com).
2. `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)`.
3. `Entorno de ejecución → Ejecutar todas`. La descarga del dataset y el entrenamiento
   (15 épocas por defecto) toman unos 15-30 minutos en GPU T4.
4. Descarga los 4 archivos generados (`transformer_en_es.pth`, `vocab_en.json`, `vocab_es.json`,
   `model_config.json`) desde el panel de archivos de Colab (o guárdalos en tu Drive).
5. Sube `mt_en_es_inferencia.ipynb` como notebook aparte, sube esos 4 archivos en su panel de
   archivos, y ejecútalo para traducir oraciones nuevas.

## Sobre los entregables del curso

1. **Código de extracción y entrenamiento** → `mt_en_es_transformer.ipynb` (secciones 1-6).
2. **Reporte** (creación del set de entrenamiento + resultados en ≥100 ejemplos de test) →
   `mt_en_es_transformer.ipynb` secciones 3, 8 y 10 (reporte narrativo), más `resultados_test.csv`.
3. **Pesos del modelo** → `transformer_en_es.pth` (se genera al ejecutar el notebook 1 en Colab).
4. **Notebook de inferencia** → `mt_en_es_inferencia.ipynb`.
5. **Presentación** → una vez que tengas los resultados reales (BLEU, ejemplos) de tu corrida en
   Colab, dime los números y armamos las diapositivas de ~12 minutos con ellos.

## Nota importante

Estos notebooks no se han ejecutado aquí (este entorno no tiene GPU ni PyTorch instalable por
límites del sistema de archivos de Windows). El código fue verificado sintácticamente y revisado
a mano (formas de tensores, máscaras de atención, máscara causal, teacher forcing), pero **debes
correrlo tú en Colab** para generar los pesos reales y completar la sección 10.3 del reporte con
tus números de BLEU y ejemplos concretos. Si algo falla al ejecutarlo, pégame el error y lo
arreglamos.
