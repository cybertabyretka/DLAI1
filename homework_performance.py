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
def measure_time_cpu(func, *args, **kwargs) -> float:
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


def measure_time_gpu(func, *args, **kwargs) -> float:
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
        print(f'[CPU] {tensor.shape}: {cpu_time}')
        cpu_times.append(cpu_time)
        if torch.cuda.is_available():
            gpu_tensor = tensor.to('cuda')
            gpu_time = measure_time_gpu(func, gpu_tensor)
            print(f'[GPU] {tensor.shape}: {gpu_time}')
            gpu_times.append(gpu_time)
        else:
            print(f'[GPU] CUDA не найдена')
    return cpu_times, gpu_times


# 3.3 Сравнение операций
result_times = {}
for func_name in funcs:
    print(func_name)
    cpu_times, gpu_times = measure_times(funcs[func_name], tensors)
    cpu_times_sum = sum(cpu_times)
    gpu_times_sum = sum(gpu_times) if len(gpu_times) > 0 else 0
    result_times[func_name] = (cpu_times_sum, gpu_times_sum)
print(f'{"Операция":<36} | {"CPU сумм.(мс)":<18} | {"GPU сумм.(мс)":<18} | {"Ускорение":<18}')
for func_name in result_times:
    cpu_time = result_times[func_name][0]
    gpu_time = result_times[func_name][1]
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f'{func_name:<36} | {cpu_time:<18.2f} | {gpu_time:<18.2f} | {speedup:<18.2f}')
