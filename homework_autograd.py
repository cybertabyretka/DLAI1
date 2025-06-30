import torch


# 2.1 Простые вычисления с градиентами
print(f'{"-"*70}\n2.1 Простые вычисления с градиентами\n{"-"*70}')
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(6., requires_grad=True)
z = torch.tensor(10., requires_grad=True)
print(f'Стартовые тензоры:\nx:\n{x}\ny:\n{y}\nz:\n{z}')
f = x**2 + y**2 + z**2 + 2*x*y*z
print(f'Значение функции:\n{f}')
f.backward()
# Аналитически вычисленные частные производные
df_dx = 2*x + 2*y*z
df_dy = 2*y + 2*x*z
df_dz = 2*z + 2*x*y
print('Проверка градиентов, вычисленных автоматически и аналитически:')
assert x.grad.item() == df_dx, 'Градиент тензора x вычислен неправильно'
print(f'{x.grad.item()} == {df_dx}')
assert y.grad.item() == df_dy, 'Градиент тензора y вычислен неправильно'
print(f'{y.grad.item()} == {df_dy}')
assert z.grad.item() == df_dz, 'Градиент тензора z вычислен неправильно'
print(f'{z.grad.item()} == {df_dz}')


# 2.2 Градиент функции потерь
print(f'{"-"*70}\n2.2 Градиент функции потерь\n{"-"*70}')


def mse(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Функция для вычисления MSE.
    :param x: Тензор объектов
    :param w: Тензор весов
    :param b: Свободный член
    :param y_true: Тензор истинных значений
    :return: MSE
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f'x должен быть torch.Tensor, получен {type(x)}')
    if not isinstance(w, torch.Tensor):
        raise TypeError(f'w должен быть torch.Tensor, получен {type(w)}')
    if not isinstance(b, torch.Tensor):
        raise TypeError(f'b должен быть torch.Tensor, получен {type(b)}')
    if not isinstance(y_true, torch.Tensor):
        raise TypeError(f'y_true должен быть torch.Tensor, получен {type(y_true)}')

    if x.shape != y_true.shape:
        raise ValueError(
            f'Размерности тензоров x и y_true не совпадают: x: {x.shape} vs y_true: {y_true.shape}'
        )
    if not w.requires_grad or not b.requires_grad:
        raise AttributeError(
            f'У переменных w и b параметр requires_grad должен быть True. w: {w.requires_grad}, b: {b.requires_grad}'
        )

    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()
    y_pred = w * x + b
    error = torch.mean((y_pred - y_true) ** 2)
    error.backward()
    if w.dim() == 0:
        print(f'Градиент w:\n{w.grad.item()}')
    else:
        print(f'Градиент w:\n{w.grad}')
    if b.dim() == 0:
        print(f'Градиент b:\n{b.grad.item()}')
    else:
        print(f'Градиент b:\n{b.grad}')
    return error


x = torch.tensor([1, 2, 3, 4])
y_true = torch.tensor([2., 4., 6., 8.])
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

try:
    mse_error = mse(x, w, b, y_true)
    print(f'Ошибка mse:\n{mse_error.item()}')
except ValueError as e:
    print(f'При вычислении mse произошла ошибка: {e}')
except AttributeError as e:
    print(f'При вычислении mse произошла ошибка: {e}')
except TypeError as e:
    print(f'При вычислении mse произошла ошибка: {e}')


# 2.3 Цепное правило
print(f'{"-"*70}\n2.3 Цепное правило\n{"-"*70}')


def f(x: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет значение формулы sin(x**2 + 1).
    :param x: Параметр формулы
    :return: Значение формулы
    """
    return torch.sin(x**2 + 1)


x = torch.tensor(2., requires_grad=True)
y = f(x)

df_dx = 2 * x * torch.cos(x**2 + 1)
auto_grad = torch.autograd.grad(y, x)[0]
print('Проверка градиентов, вычисленных автоматически и аналитически:')
assert auto_grad == df_dx.item(), 'Градиент тензора x вычислен неправильно'
print(f'{auto_grad} == {df_dx}')