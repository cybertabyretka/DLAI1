import torch
import time
from typing import Callable

# 3.1 Подготовка данных
tensors = [
    torch.rand(64, 1024, 1024),
    torch.rand(128, 512, 512),
    torch.rand(256, 256, 256)
]
print('Тензоры проинициализированы')
funcs = {
    'Матричное умножение': lambda m: torch.matmul(m, m.mT),
    'Поэлементное сложение': lambda m: m + m,
    'Поэлементное умножение': lambda m: m * m,
    'Транспонирование': lambda m: m.mT,
    'Вычисление суммы всех элементов': lambda m: torch.sum(m)
}


# 3.2 Функция измерения времени
def measure_time_cpu(func: Callable, *args, **kwargs) -> float:
    """
    Вычисляет время выполнения функции на CPU.
    :param func: Функция, время выполнения которой нужно узнать
    :param args: Аргументы функции
    :return: Время выполнения переданной функции
    """
    start_time = time.time()
    func(*args, **kwargs)
    cpu_time = (time.time() - start_time) * 1000
    return cpu_time


def measure_time_gpu(func: Callable, *args, **kwargs) -> float:
    """
    Вычисляет время выполнения функции на GPU.
    :param func: Функция, время выполнения которой нужно узнать
    :param args: Аргументы функции
    :return: Время выполнения переданной функции
    """
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        gpu_time = start.elapsed_time(end)
    else:
        raise RuntimeError('CUDA не найдена')
    return gpu_time


def measure_times(func: Callable, tensors: list[torch.Tensor]) -> tuple[list, list]:
    """
    Рассчитывает время выполнения функции на переданных тензорах. Считает сразу и на CPU, и на GPU.
    :param func: Функция, время выполнения которой нужно узнать
    :param tensors: Тензоры, на которых будет запускаться функция
    :return: Кортеж из двух списков, первый список содержит время на CPU, второй время на GPU
    """
    cpu_times = []
    gpu_times = []
    for tensor in tensors:
        cpu_time = measure_time_cpu(func, tensor)
        print(f'[CPU] {tensor.shape}: {cpu_time:.2f} мс')
        cpu_times.append(cpu_time)
        if torch.cuda.is_available():
            gpu_tensor = tensor.to('cuda')
            gpu_time = measure_time_gpu(func, gpu_tensor)
            print(f'[GPU] {tensor.shape}: {gpu_time:.2f} мс')
            gpu_times.append(gpu_time)
        else:
            print(f'[GPU] CUDA не найдена')
            gpu_times.append(0.0)
    return cpu_times, gpu_times


# 3.3 Сравнение операций
print(f'{"-"*70}\n3.3 Сравнение операций\n{"-"*70}')
result_times = {}
for func_name in funcs:
    print(f"\nИзмерение для операции: {func_name}")
    cpu_times, gpu_times = measure_times(funcs[func_name], tensors)
    op_results = []
    for i, tensor in enumerate(tensors):
        shape_str = str(tensor.numel())
        op_results.append((shape_str, cpu_times[i], gpu_times[i]))
    result_times[func_name] = op_results
print()
print(f"{'Операция':<36} | {'Размер тензора':<20} | {'CPU (мс)':<12} | {'GPU (мс)':<12} | {'Ускорение':<12}")
for func_name, results in result_times.items():
    for shape_str, cpu_time, gpu_time in results:
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"{func_name:<36} | {shape_str:<20} | {cpu_time:<12.2f} | {gpu_time:<12.2f} | {speedup:<12.2f}")


# 3.4 Анализ результатов
"""
Анализ результатов: 
1. На GPU наибольшее ускорение получают следующие операции: Поэлементное умножение, 
Поэлементное умножение, Вычисление суммы всех элементов. Это говорит о том, что эти операции очень хорошо 
распараллеливаются.
2. Таких операций, которые значительно ухудшили время не было. Но была операция с нулевым ускорением: Транспонирование.
Эта операция требует практически полного переупорядочивания данных в памяти, поэтому накладные расходы GPU превышают 
выгоду.
3. Размер матриц по разному влияет на ускорение. При Матричном умножении и Поэлементном умножении ускорение обратно 
попорционально размеру тензора. Это можно объяснить тем, что при малых размерах, накладные
расходы GPU не дают получить значительную выгоду и только при увеличении размерности тензоров накладные расходы 
начинают играть меньшую роль.
4. При передаче данных между CPU и GPU происходит:
    4.1. Инициализация передачи данных
        4.1.1. CPU подготавливает данные в оперативной памяти (RAM).
        4.1.2. GPU имеет свою собственную память (VRAM), и данные должны быть скопированы туда для обработки.
    4.2. Дискретные GPU (NVIDIA, AMD) подключаются через Через шину PCI Express.
         Скорость передачи зависит от версии PCIe:
            PCIe 3.0 x16: ~16 ГБ/с
            PCIe 4.0 x16: ~32 ГБ/с
            PCIe 5.0 x16: ~64 ГБ/с
         Данные копируются из RAM → VRAM через DMA (прямой доступ к памяти).
    4.3. Роль драйверов и API
         Драйвер GPU управляет передачей данных.
         API (CUDA, OpenCL, Vulkan, DirectX) предоставляют функции для:
            Выделения памяти на GPU (cudaMalloc, clCreateBuffer).
            Копирования данных (cudaMemcpy, clEnqueueWriteBuffer).
            Синхронизации между CPU и GPU.
    4.4. Задержки и оптимизации
         Латентность: Передача через PCIe добавляет задержку (~микросекунды).
         Оптимизации:
            Пакетная передача (меньше вызовов API)
            Асинхронные копии (передача данных параллельно с вычислениями)
            Совместная память
            Сжатие данных
"""