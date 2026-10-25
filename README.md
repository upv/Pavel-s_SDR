# Pavel-s_SDR

Эксперименты с SDR-платами: ADALM-Pluto, ANTSDR E310/E316, USRP B210, KrakenSDR,
Orange Pi AI Pro (Ascend). Синхронизация по последовательностям Задова–Чу,
оценка STO/CFO, OFDM, приём 5G NR PUSCH, полный стек 5G NR на Sionna Research Kit.

| Папка | Что внутри |
|---|---|
| [`srk_e316_ue/`](srk_e316_ue/README.md) | 5G NR SA на Sionna Research Kit: gNB на U220, OAI nrUE на ANTSDR E316 (SISO и 2x2), журнал экспериментов |
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
