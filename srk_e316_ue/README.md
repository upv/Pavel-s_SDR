# 5G NR SA на Sionna Research Kit: gNB на U220, UE на ANTSDR E316

Полный стек 5G NR (ядро OAI CN5G, gNB, nrUE) из [NVIDIA Sionna Research Kit](https://github.com/NVlabs/sionna-rk) v1.3.3
на DGX Spark. gNB работает на ANTSDR U220 (клон B210), UE (OAI nrUE) на ANTSDR E316 с UHD-прошивкой MicroPhase,
обе платы подключены к одному хосту. Проверено по эфиру в SISO и 2x2 MIMO, журнал в [EXPERIMENTS.md](EXPERIMENTS.md).

| Файл | Что это |
|---|---|
| `patches/oai_usrp_lib_antsdr.patch` | Патч `radio/USRP/usrp_lib.cpp` OAI 2025.w34 (см. ниже) |
| `patches/gnb_24prbs_2x2.conf.patch` | gNB 24 PRB band 78 → 2x2: 2 порта PDSCH/PUSCH, CSI-RS, SRS, `nb_tx/nb_rx = 2` |
| `docker/ue/Dockerfile` | `oai-nr-ue-antsdr`: SRK nrUE + antsdr_uhd + патченый модуль USRP |
| `docker/gnb/Dockerfile` | `oai-gnb-cuda-txrx`: SRK gNB + патченый модуль USRP |
| `build_images.sh` | Сборка модуля в `ran-build-cuda` и обоих образов |
| `srk_config/` | `.env` для `config/b200` (SISO) и `config/b200_2x2`, `docker-compose.override.yaml` |
| `tools/ssb_capture.py` | Поиск SSB (PSS/SSS, PCI, CFO) в записи E316 (pyadi-iio или файл `.cf32`) |
| `tools/ssb_level.py` | Уровень PSS по нескольким записям (сравнение антенн/портов) |

## Что меняет патч `usrp_lib.cpp`

1. Тип устройства `ant` (antsdr_uhd) обрабатывается как B210, но без `send/recv_frame_size=7680`:
   по 1 GbE без jumbo-кадров они не проходят. Без этого OAI падает с segfault (нет калибровочной таблицы).
2. `OAI_USRP_RX_ANTENNA` — антенна приёма (например `TX/RX`, когда антенна стоит только на порту TX/RX).
3. `OAI_USRP_RX_OTW` / `OAI_USRP_TX_OTW` — формат на проводе. Для E316 2x2 нужно `sc8`/`sc12`:
   2 × 15.36 MSps в `sc16` ≈ 1.03 Гбит/с, не помещается в 1 GbE; `sc12` на приём прошивка E316 не поддерживает.
4. Ветка 11.52 MSps для B2xx (3/4 sampling). Для 24 PRB не пригодилась: в OAI нет iDFT на 384 точки.

## Как поднять

1. SRK по [quickstart](https://nvlabs.github.io/sionna/rk/quickstart.html): `build-oai-images.sh`, `generate-configs.sh`,
   плагины. **Не запускать `scripts/quickstart-cn5g.sh` v1.3.3**: файла `patches/oai-cn5g.patch` нет, и
   `patch-oai-cn5g.sh` выполняет `rsync -av / ext/oai-cn5g-fed/`, копируя весь диск. Ядро берётся готовыми
   arm64-образами `oaisoftwarealliance/oai-{amf,smf,upf}:v2.1.10`.
2. E316: SD-карта FAT32 с файлами из `e310v2.zip` релиза [antsdr_uhd v1.0](https://github.com/MicroPhase/antsdr_uhd/releases/tag/v1.0),
   кабель Cat5e+ (нужен 1000M), адрес платы 192.168.1.10, на хосте 192.168.1.100/24.
3. antsdr_uhd на хосте (`host/README.md` репозитория MicroPhase) в `$HOME/opt/antsdr-uhd`, нужен `libusb-1.0-0-dev`.
4. `OAI_DIR=.../sionna-rk/ext/openairinterface5g ./build_images.sh`
5. Скопировать `srk_config/*.env` в `sionna-rk/config/<b200|b200_2x2>/.env` (вписать `USRP_SERIAL` U220),
   override в `config/common/` (вписать каталог образов UHD для U220), 2x2-конфиг gNB — патчем.
6. Для 2x2: `sudo sysctl -w net.core.rmem_max=62500000 net.core.wmem_max=62500000` (до перезагрузки).
7. Запуск: `./scripts/start_system.sh b200_2x2`, затем
   `cd config/common && docker compose --env-file ../b200_2x2/.env up -d oai-nr-ue`.

Особенности E316: частота платы смещена примерно на −37.5 кГц (~10 ppm) относительно U220, поэтому у UE стоит
`--initial-fo -37500 --ue-fo-compensation`. U220 грузит FPGA-образ из UHD 4.9 (образ UHD 4.6 не работает),
в контейнер gNB он монтируется через override.
