# Pavel-s_SDR

Эксперименты с SDR-платами: ADALM-Pluto, ANTSDR E310/E316, USRP B210, KrakenSDR,
Orange Pi AI Pro (Ascend). Синхронизация по последовательностям Задова–Чу,
оценка STO/CFO, OFDM, приём 5G NR PUSCH.

| Папка | Что внутри |
|---|---|
| [`nr_pusch_e316/`](nr_pusch_e316/README.md) | Приёмник и передатчик 5G NR PUSCH из симулятора WirelessNN для ANTSDR E316 |
| [`Code_TT/`](Code_TT) | Задачи по синхронизации: корреляция ZC (`zc_rs.py`), оценка STO и CFO по CP (`zc_part2.py`) |
| [`tests/`](tests) | Тесты pytest для `Code_TT` |
| [`archive/`](archive) | Ранние скрипты и ноутбуки (Pluto, E310, Kraken, Ascend, GNU Radio), записи сигналов |

## Запуск тестов

```
pip install -r Code_TT/requirements.txt
pytest -v tests
```

## Лицензия

MIT, см. [LICENSE](LICENSE).
