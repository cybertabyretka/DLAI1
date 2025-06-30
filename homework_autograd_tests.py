import pytest
import torch
from homework_autograd import mse


def test_mse_correct_calculation() -> None:
    """
    Проверка корректности расчета MSE и градиентов.
    :return: None
    """
    x = torch.tensor([1.0, 2.0, 3.0])
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    y_true = torch.tensor([3.0, 5.0, 7.0])

    error = mse(x, w, b, y_true)

    expected_error = torch.mean((w * x + b - y_true) ** 2)
    assert torch.isclose(error, expected_error)

    expected_w_grad = torch.mean(2 * (w * x + b - y_true) * x).item()
    expected_b_grad = torch.mean(2 * (w * x + b - y_true)).item()
    assert torch.isclose(torch.tensor(w.grad.item()), torch.tensor(expected_w_grad))
    assert torch.isclose(torch.tensor(b.grad.item()), torch.tensor(expected_b_grad))


def test_mse_shape_mismatch() -> None:
    """
    Проверка вызова исключения при несовпадении размерностей x и y_true.
    :return: None
    """
    x = torch.tensor([1.0, 2.0])
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    y_true = torch.tensor([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Размерности тензоров x и y_true не совпадают"):
        mse(x, w, b, y_true)


def test_mse_requires_grad_false() -> None:
    """
    Проверка вызова исключения, если requires_grad=False у w или b.
    :return: None
    """
    x = torch.tensor([1.0])
    y_true = torch.tensor([1.0])
    w = torch.tensor(1.0)
    b = torch.tensor(0.0, requires_grad=True)
    with pytest.raises(AttributeError, match="У переменных w и b параметр requires_grad должен быть True"):
        mse(x, w, b, y_true)
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0)
    with pytest.raises(AttributeError, match="У переменных w и b параметр requires_grad должен быть True"):
        mse(x, w, b, y_true)


def test_mse_zero_error() -> None:
    """
    Проверка случая, когда предсказания совпадают с истинными значениями (ошибка = 0).
    :return: None
    """
    x = torch.tensor([1.0, 2.0])
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    y_true = w * x + b

    error = mse(x, w, b, y_true)
    assert torch.isclose(error, torch.tensor(0.0))

    assert torch.isclose(w.grad, torch.tensor(0.0))
    assert torch.isclose(b.grad, torch.tensor(0.0))


def test_mse_batch_processing() -> None:
    """
    Проверка обработки многомерных данных.
    :return: None
    """
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    w = torch.tensor([2.0, 3.0], requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    y_true = torch.tensor([[3.0, 7.0], [7.0, 13.0]])
    error = mse(x, w, b, y_true)
    y_pred = w * x + b
    expected_error = torch.mean((y_pred - y_true) ** 2)
    assert torch.isclose(error, expected_error)
    expected_w_grad = torch.mean(2 * (y_pred - y_true) * x, dim=0)
    expected_b_grad = torch.mean(2 * (y_pred - y_true))
    assert torch.allclose(w.grad, expected_w_grad)
    assert torch.isclose(b.grad, expected_b_grad)