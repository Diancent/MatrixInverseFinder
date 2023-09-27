import numpy as np

# Функція для обчислення оберненої матриці методом Гаусса
def inverse_matrix_gauss(matrix):
    n = len(matrix)
    augmented_matrix = np.hstack((matrix, np.identity(n)))

    # Прямий хід методу Гаусса
    for i in range(n):
        pivot_row = i
        for j in range(i + 1, n):
            if abs(augmented_matrix[j, i]) > abs(augmented_matrix[pivot_row, i]):
                pivot_row = j
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]

        pivot_element = augmented_matrix[i, i]
        augmented_matrix[i] /= pivot_element

        for j in range(i + 1, n):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]

    # Зворотний хід методу Гаусса
    for i in range(n - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]

    # Видаляємо оригінальну матрицю і залишаємо обернену
    inverse = augmented_matrix[:, n:]

    return inverse

# Функція для обчислення норми Фробеніуса матриці
def frobenius_norm(matrix):
    return np.linalg.norm(matrix, 'fro')

# Функція для обчислення детермінанту
def calculate_determinant(matrix):
    n = len(matrix)
    det = 1.0

    # Копія матриці для виконання відшаровування без змін в початковій матриці
    mat_copy = np.copy(matrix)

    for i in range(n):
        # Шукаємо перший ненульовий елемент у поточному стовпці
        pivot_index = -1
        for j in range(i, n):
            if mat_copy[j, i] != 0:
                pivot_index = j
                break

        if pivot_index == -1:
            # Якщо не знайдено ненульового елемента в стовпці, то визначник дорівнює нулю
            return 0.0

        # Обмін рядків, якщо потрібно, для отримання ненульового діагонального елементу
        if pivot_index != i:
            mat_copy[[i, pivot_index]] = mat_copy[[pivot_index, i]]
            det *= -1  # Змінюємо знак визначника при обміні рядків

        # Обчислюємо визначник зі змінами в знаку
        det *= mat_copy[i, i]

        # Відшаровуємо матрицю, роблячи нульовими елементи під діагоналлю
        for j in range(i + 1, n):
            factor = mat_copy[j, i] / mat_copy[i, i]
            mat_copy[j, i:] -= factor * mat_copy[i, i:]
    return det

def calculate_determinant_from_file(input_filename):
    try:
        with open(input_filename, 'r') as file:
            lines = file.readlines()
            input_matrix = [list(map(float, line.strip().split())) for line in lines]
        input_matrix = np.array(input_matrix)  # Перетворення в NumPy-масив
        determinant = calculate_determinant(input_matrix)
        return determinant
    except Exception as e:
        raise ValueError("Помилка при обчисленні детермінанту з файлу:", str(e))

# Функція для зчитування матриці з файлу
def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        matrix = []
        for line in lines:
            row = [float(x) for x in line.strip().split()]
            matrix.append(row)
    return np.array(matrix)

# Функція для запису матриці у файл
def write_matrix_to_file(matrix, filename, determinant, error, decimal_places=2):
    with open(filename, 'w') as file:
        for row in matrix:
            formatted_row = ['{:.{}f}'.format(x, decimal_places) for x in row]
            file.write(' '.join(formatted_row) + '\n')
        file.write(f'\nДетермінант: {determinant:.2f}\n')
        file.write(f'\nПохибка: {error}')

# Використання
if __name__ == "__main__":
    input_filename = "input_matrix.txt"
    output_filename = "inverse_matrix.txt"

    # Зчитуємо матрицю з файлу
    A = read_matrix_from_file(input_filename)

    # Обчислюємо обернену матрицю
    A_inv = inverse_matrix_gauss(A)

    # Детермінант
    determinant = calculate_determinant_from_file(input_filename)

    # Обчислюємо похибку за допомогою норми Фробеніуса
    error = frobenius_norm(np.dot(A, A_inv) - np.identity(len(A)))

    # Записуємо обернену матрицю у файл
    write_matrix_to_file(A_inv, output_filename, determinant, error)

    print("Обернена матриця:")
    print(A_inv)
    print("Похибка за допомогою норми Фробеніуса:", frobenius_norm(np.dot(A, A_inv) - np.identity(len(A))))
    print("Обернена матриця була записана у файл", output_filename)
