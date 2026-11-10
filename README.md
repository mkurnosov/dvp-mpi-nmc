# Distributed Pipeline for Neural Network-Based Video Processing on a Cluster of NeuroMatrix Processors

Реализация на MPI конвейера нейросетевой обработки видеофайлов на кластере процессоров NeuroMatrix (шаблон "производитель-потребитель")
```shell
|-- data                    // Тестовый входной mp4 файл
|-- dvp_mpi_dynamic_nmc     // Версия с динамическим распределением блоков
|-- dvp_mpi_static_nmc      // Версия со статическим назначением блоков рабочим процессам
|-- dvp_nmc                 // Последовательная версия с использованием одного SoC NM6408 (multi-unit mode)
`-- model                   // Модели для инференса на NM6408
    |-- yolo_v8n
    `-- yolo_v8s
```
