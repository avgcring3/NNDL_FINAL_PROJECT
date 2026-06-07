# Условия Capstone Pitch и соответствие RideFlow NN

## Точные условия из PDF курса

### Оценивание

- Base Score: **8 points**.
- Bonus: **+2 points** за real-world operational data.
- Maximum: **10 points**.

Real-world operational data определяется как:

- organization-owned data;
- production-like data;
- public operational datasets;
- не toy/academic samples.

## Обязательная структура pitch

### 1. Problem Statement

- What business problem are we solving?
- Why does it matter now?

### 2. Data

- What data will you use?
- Why is it reliable / representative?

### 3. Modeling Approach

- Proposed architecture / method.
- Why this approach for this problem?

### 4. KPI & Success

- Primary success metric.
- Expected business impact.

## Дополнительные ожидания из материалов курса

- четко определить prediction task и target;
- показать architecture sketch;
- реализовать воспроизводимый data pipeline;
- использовать train/validation/test split;
- вести loss logging;
- определить quantitative metrics и baselines;
- проверить overfitting/underfitting через train/validation curves;
- показать qualitative outputs и error plots;
- держать первую версию небольшой и стабильной;
- удалить dead code и объяснить interfaces.

## Соответствие RideFlow NN

| Требование | Реализация |
|---|---|
| Business problem | несоответствие спроса и расположения водителей; непонятная будущая цена |
| Data | hourly zone demand, 12 Moscow zones, 60 days |
| Neural network | Transformer Encoder в PyTorch |
| Sequence modeling | sliding windows из 48 часов |
| Attention | 4-head self-attention |
| Position information | sinusoidal positional encoding |
| Categorical context | zone embedding |
| Regularization | dropout 0.1 |
| Custom loss | weighted MAE для high-demand targets |
| Validation | chronological 70/15/15 split |
| Baselines | last value, same hour yesterday |
| Metrics | MAE, RMSE, sMAPE, underforecast rate, Peak MAE |
| Logging | training_curve.csv |
| Checkpoint | transformer_model.pt |
| Qualitative output | forecast_sample.png и dashboard |
| Business demo | адреса, маршрут, будущая дата, цена, коэффициенты |

## Честный статус бонуса +2

Текущий включенный Moscow dataset является synthetic operational-style dataset. Он подходит для демонстрации полного pipeline, но его нельзя выдавать за реальные заказы.

Для уверенного получения бонуса `+2` необходимо до защиты:

1. обучить pipeline на public operational NYC TLC data через режим `--mode tlc`; или
2. получить реальные логи заказов партнера и загрузить их через CSV adapter.

Если этого не сделать, на презентации необходимо говорить:

> Current Moscow data is reproducible synthetic operational-style data. The pipeline supports real operational data, but the included demo does not claim the +2 real-world-data bonus.

## Почему RideFlow NN лучше соответствует курсу, чем SubSaver

| SubSaver | RideFlow NN |
|---|---|
| rule-based scoring | обучаемая нейронная сеть |
| готовый OpenAI API | собственный Transformer в PyTorch |
| нет train/validation/test | chronological split |
| нет baseline comparison | два naive baseline |
| нет model metrics | пять метрик |
| автоматизация как основной результат | neural forecasting как основной результат |

