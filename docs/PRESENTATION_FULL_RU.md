# RideFlow NN — полное описание презентации

## 1. Короткое позиционирование проекта

**RideFlow NN** — система прогнозирования почасового спроса на такси по зонам Москвы и объяснимой оценки будущей стоимости поездки.

Пользователь задает точку A, точку B, дату, время и класс тарифа. Система:

1. находит адреса;
2. строит автомобильный маршрут;
3. получает прогноз погоды;
4. получает прогноз спроса от обученной нейронной сети;
5. рассчитывает коэффициенты спроса, погоды и трафика/времени;
6. показывает итоговую оценку цены и объяснение каждого коэффициента.

## 2. Почему первоначальный SubSaver был заменен

Первоначальный pitch SubSaver решал понятную бизнес-проблему, но его modeling approach состоял преимущественно из:

- rule-based scoring;
- автоматизации в n8n;
- вызова готового OpenAI API;
- уведомлений и рекомендаций.

В таком варианте отсутствовало собственное обучение нейронной сети. Поэтому проект не демонстрировал ключевые темы NNDL: подготовку последовательностей, обучение модели, loss function, train/validation/test split, baselines и количественную оценку качества.

RideFlow NN закрывает это требование:

- модель реально обучается в PyTorch;
- используется Transformer Encoder;
- реализованы positional encoding, multi-head self-attention и zone embeddings;
- применяется custom weighted MAE loss;
- используются временной split и baselines;
- сохраняются метрики, loss curve, прогнозы и веса модели.

---

## 3. Рекомендуемая структура презентации

### Слайд 1. Title

**На слайде**

- RideFlow NN
- Neural Taxi Demand Forecasting and Explainable Future Fare Estimation
- Moscow demo, Transformer model, 14-day forecast

**Что сказать**

> RideFlow NN прогнозирует, где и когда появится спрос на такси, а затем использует этот прогноз для объяснимой оценки стоимости будущей поездки. Это не просто интерфейс или набор правил: внутри обучается Transformer для временных рядов.

---

### Слайд 2. Problem Statement

**Бизнес-проблема**

- спрос на такси меняется по району и часу;
- водители могут находиться не там, где появятся заказы;
- пассажиры получают долгую подачу и высокий динамический коэффициент;
- оператор теряет завершенные поездки, а водители простаивают.

**Почему это важно сейчас**

- спрос зависит от часа, дня недели, погоды, событий и района;
- статические средние плохо работают в пиковые часы;
- оператору нужен прогноз до появления спроса, а пользователю — понятная оценка будущей цены.

**Что сказать**

> Мы решаем две связанные задачи. Первая — заранее прогнозировать количество заказов по зоне и часу. Вторая — использовать прогноз спроса вместе с маршрутом, погодой и временем для оценки будущей цены поездки.

---

### Слайд 3. Product Demo

**На слайде**

- ввод точки A и точки B;
- выбор будущей даты и времени;
- выбор тарифа;
- маршрут на карте;
- итоговая цена;
- breakdown коэффициентов.

**Что показать в live demo**

1. Ввести `Шаболовская`.
2. Ввести `Кузьминки`.
3. Выбрать обычное время и показать умеренный коэффициент.
4. Переключить время на будний вечерний пик.
5. Показать рост traffic/time coefficient и итоговой цены.
6. Открыть вкладку Future Forecast.

**Что сказать**

> Интерфейс показывает не только итоговую цену. Он показывает базовую стоимость и вклад спроса, погоды и трафика/времени. Значение x1.00 означает нормальные условия, а не пропущенный расчет.

---

### Слайд 4. Data

**Текущий включенный датасет**

- воспроизводимые synthetic operational-style данные;
- 12 московских зон;
- почасовой спрос;
- 60 дней истории;
- паттерны утренних и вечерних пиков;
- отдельные паттерны аэропортов, выходных и событий;
- даты автоматически заканчиваются на текущем московском часу.

**Размер текущего набора**

- 17 280 почасовых строк;
- 16 704 sliding-window samples;
- 48 часов истории на входе;
- прогноз следующего часа при обучении;
- рекурсивный прогноз на 336 часов / 14 дней.

**Честное ограничение**

Включенный Moscow dataset является synthetic и не дает бонус `+2` за real-world operational data.

**Как получить бонус**

- заменить demo CSV на реальные trip-level логи партнера;
- либо обучить pipeline через включенный NYC TLC parquet adapter;
- сохранить те же модель, split и метрики.

**Что сказать**

> Для готового автономного демо мы используем воспроизводимые данные с операционными паттернами Москвы. Мы не называем их реальными заказами. Код уже поддерживает CSV и NYC TLC parquet, поэтому для бонусных баллов достаточно заменить источник на реальный operational dataset и переобучить модель.

---

### Слайд 5. Prediction Task

**Формулировка**

- объект: зона Москвы;
- временной шаг: один час;
- вход: последние 48 значений спроса;
- target: спрос следующего часа;
- дополнительный признак: идентификатор зоны;
- deployment output: прогноз по зонам на 14 дней.

**Sliding window**

```text
[demand(t-47), ..., demand(t)] -> demand(t+1)
```

**Что сказать**

> Мы преобразуем каждый временной ряд зоны в sliding windows. Это напрямую соответствует теме time-series prediction из курса. Разделение выполняется по времени, поэтому модель не видит будущее во время обучения.

---

### Слайд 6. Modeling Approach

**Основная модель: DemandTransformer**

- value projection: `Linear(1 -> 32)`;
- zone embedding;
- sinusoidal positional encoding;
- Transformer Encoder;
- 4 attention heads;
- feed-forward dimension `128`;
- GELU activation;
- dropout `0.1`;
- LayerNorm;
- residual forecast head.

**Почему Transformer**

- спрос является последовательностью;
- self-attention может учитывать разные часы предыдущих двух суток;
- positional encoding сохраняет порядок;
- zone embedding учитывает различия районов;
- residual head стабилизирует обучение относительно последнего известного спроса.

**Что сказать**

> Transformer получает последовательность спроса, добавляет embedding зоны и positional encoding. Multi-head self-attention учится выбирать важные часы истории. Финальная голова прогнозирует residual correction относительно последнего известного значения.

---

### Слайд 7. Course Concepts Used

**Темы курса, реализованные в проекте**

- neural network foundations и backpropagation;
- time-series sliding windows;
- embeddings;
- positional encoding;
- multi-head self-attention;
- Transformer Encoder;
- residual connection;
- LayerNorm;
- dropout regularization;
- custom weighted loss;
- train/validation/test time split;
- baselines;
- reproducible training with fixed seed;
- quantitative validation and loss curve.

**Что сказать**

> Проект использует не одну отдельную тему, а полный neural-network workflow: от подготовки последовательностей и custom loss до Transformer, baselines и временной валидации.

---

### Слайд 8. Custom Loss and Training

**Weighted MAE**

Обычная MAE считает все ошибки одинаковыми. Для такси ошибка в пиковый час дороже, потому что приводит к нехватке автомобилей.

В проекте:

- определяется 75-й перцентиль target demand внутри batch;
- значения выше порога получают вес `2.0`;
- модель сильнее штрафуется за ошибку на высоком спросе.

```text
Weighted MAE = mean(weight(target) * abs(prediction - target))
```

**Training setup**

- PyTorch;
- AdamW;
- learning rate `0.001`;
- batch size `256`;
- 5 epochs;
- fixed seed `42`;
- log1p transform demand.

---

### Слайд 9. Validation and Baselines

**Корректная проверка**

- 70% train;
- 15% validation;
- 15% test;
- split выполняется по timestamp;
- будущее не попадает в train.

**Baselines**

- last observed demand;
- same hour yesterday.

**Метрики**

- MAE;
- RMSE;
- sMAPE;
- underforecast rate;
- peak-hour MAE.

**Что сказать**

> Нейронную сеть нельзя оценивать без baseline. Мы сравниваем Transformer с последним известным спросом и тем же часом предыдущего дня.

---

### Слайд 10. Results

| Model | MAE | RMSE | sMAPE | Underforecast Rate | Peak MAE |
|---|---:|---:|---:|---:|---:|
| Transformer | **3.920** | **5.624** | **0.0407** | **0.3847** | **5.282** |
| Last value baseline | 4.568 | 6.581 | 0.0472 | 0.4717 | 6.710 |
| Same hour yesterday | 5.174 | 7.268 | 0.0533 | 0.4840 | 7.699 |

**Улучшение Transformer**

- MAE лучше last-value baseline примерно на **14.2%**;
- MAE лучше day-ago baseline примерно на **24.2%**;
- Peak MAE лучше day-ago baseline примерно на **31.4%**;
- underforecast rate ниже day-ago baseline примерно на **9.9 percentage points**.

**Что сказать**

> Главный результат — модель улучшает не только общую MAE, но и peak MAE. Для бизнеса это особенно важно, потому что именно недооценка пикового спроса приводит к потерянным поездкам.

---

### Слайд 11. Explainable Fare Estimation

**Формула**

```text
Base fare =
minimum/base tariff
+ extra distance × rub/km
+ extra duration × rub/min

Final fare =
Base fare
× Demand coefficient
× Weather coefficient
× Traffic/time coefficient
```

**Компоненты**

- route: OSRM;
- addresses: ArcGIS, Photon, Nominatim fallback;
- weather: Open-Meteo, wttr.in fallback;
- demand: Transformer forecast percentile;
- traffic/time: route speed estimate и календарные пики.

**Важно**

Это объяснимая оценка, а не официальный quote Яндекс Go. Точный динамический коэффициент агрегатора закрыт.

---

### Слайд 12. KPI & Success

**Primary success metric**

- MAE на test split;
- дополнительный приоритет: Peak MAE.

**Model success**

- Transformer должен быть лучше naive baselines;
- снижение peak-hour underforecasting;
- стабильная train/validation curve.

**Expected business impact**

- более точное позиционирование водителей;
- меньше idle time;
- меньше нехватки автомобилей в пиковых зонах;
- меньше пропущенных заказов;
- более понятная будущая цена для пассажира.

**Возможные production KPI**

- completed-trip rate;
- passenger waiting time;
- driver idle minutes;
- cancellation rate;
- revenue per active driver hour.

---

### Слайд 13. Limitations and Roadmap

**Текущие ограничения**

- Moscow demand data synthetic;
- нет официального real-time API тарифов Яндекс Go;
- OSRM не является полноценным источником live traffic;
- погода ограничена горизонтом внешнего API;
- 14-дневный рекурсивный прогноз накапливает ошибку.

**Roadmap**

1. Получить реальные логи заказов.
2. Добавить внешние признаки: погода, праздники, события, пробки.
3. Сравнить Transformer с LSTM/GRU и MLP.
4. Выполнить ablation study.
5. Перейти от районов к hexagonal spatial grid.
6. Калибровать цену по историческим фактическим поездкам.

---

### Слайд 14. Conclusion

**Финальная формулировка**

> RideFlow NN демонстрирует полный цикл neural-network проекта: данные, sliding windows, Transformer, custom loss, baselines, time-based validation, metrics и рабочий интерфейс. Модель улучшает MAE и peak MAE относительно naive approaches, а прогноз используется в понятном бизнес-сценарии планирования такси и оценки будущей стоимости поездки.

---

## 4. Рекомендуемый тайминг защиты на 7 минут

| Время | Содержание |
|---|---|
| 0:00–0:40 | проблема и business value |
| 0:40–1:20 | продукт и live demo |
| 1:20–2:10 | данные и честное ограничение |
| 2:10–3:30 | Transformer architecture |
| 3:30–4:20 | custom loss, split, baselines |
| 4:20–5:20 | результаты и KPI |
| 5:20–6:20 | price estimation и explainability |
| 6:20–7:00 | limitations, roadmap, conclusion |

## 5. Что обязательно показать преподавателю

- `src/models.py` — Transformer, positional encoding, embedding, weighted loss;
- `src/data.py` — sliding windows и time split;
- `src/train.py` — training loop, baselines, metrics, artifacts;
- `artifacts/metrics.json` — итоговые результаты;
- `reports/forecast_sample.png` — prediction vs actual;
- вкладку Future Forecast;
- вкладку Ride Price и coefficient breakdown.

