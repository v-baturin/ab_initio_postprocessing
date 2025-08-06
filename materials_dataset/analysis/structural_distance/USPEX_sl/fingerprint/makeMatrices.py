import numpy as np
from numpy.linalg import det

ELEMENT2Z = {
    'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
    'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,
    'K':19, 'Ca':20, 'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26,
    'Co':27, 'Ni':28, 'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32, 'As':33, 'Se':34,
    'Br':35, 'Kr':36, 'Rb':37, 'Sr':38, 'Y':39, 'Zr':40, 'Nb':41, 'Mo':42,
    'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48, 'In':49, 'Sn':50,
    'Sb':51, 'Te':52, 'I':53, 'Xe':54, 'Cs':55, 'Ba':56, 'La':57, 'Ce':58,
    'Pr':59, 'Nd':60, 'Pm':61, 'Sm':62, 'Eu':63, 'Gd':64, 'Tb':65, 'Dy':66,
    'Ho':67, 'Er':68, 'Tm':69, 'Yb':70, 'Lu':71, 'Hf':72, 'Ta':73, 'W':74,
    'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80, 'Tl':81, 'Pb':82,
    'Bi':83, 'Po':84, 'At':85, 'Rn':86, 'Fr':87, 'Ra':88, 'Ac':89, 'Th':90,
    'Pa':91, 'U':92, 'Np':93, 'Pu':94, 'Am':95, 'Cm':96, 'Bk':97, 'Cf':98,
    'Es':99, 'Fm':100, 'Md':101, 'No':102, 'Lr':103, 'Rf':104, 'Db':105,
    'Sg':106, 'Bh':107, 'Hs':108, 'Mt':109, 'Ds':110, 'Rg':111, 'Cn':112,
    'Nh':113, 'Fl':114, 'Mc':115, 'Lv':116, 'Ts':117, 'Og':118
}


def makeMatrices(lattice, coordinates, numIons, atomType, RmaxFing, dimension):
    """
    Ещё более оптимизированная (векторизованная) версия makeMatrices3, которая:
    - Точно воспроизводит те же результаты (N, V, dist_matrix, typ_i, typ_j),
      включая порядок строк в dist_matrix.
    - Сохраняет вложенные циклы i, j, k и quad, но убирает внутренний цикл по basic_cell.
    - Упрощена логика добавления строк в dist_matrix_list (убран дублирующийся код).

    Параметры
    ---------
    lattice : (3,3) array-like
    coordinates : (natom, 3) array-like
    numIons : 1D array-like
    atomType : 1D array-like (числа или названия элементов)
    RmaxFing : float
    dimension : int (0, 1 или 2)

    Возвращает
    ----------
    (N, V, dist_matrix, typ_i, typ_j)
    """
    # Преобразуем во float/int массивы
    lattice = np.array(lattice, dtype=float)
    coordinates = np.array(coordinates, dtype=float)
    numIons = np.array(numIons, dtype=int)

    # Переводим atomType (строки -> атомные номера)
    converted_atomType = []
    for val in atomType:
        if isinstance(val, str):
            sym = val.strip().capitalize()
            if sym in ELEMENT2Z:
                converted_atomType.append(ELEMENT2Z[sym])
            else:
                raise ValueError(f"Неизвестный атомный символ '{val}'!")
        else:
            converted_atomType.append(int(val))
    atomType = np.array(converted_atomType, dtype=int)

    # Сколько всего типов/атомов
    species = len(numIons)
    natom = np.sum(numIons)

    # Вычисляем объём
    V = abs(det(lattice))

    # Если атомов нет, вернуть пустые
    if natom == 0:
        return np.array([]), V, np.array([]), np.array([]), np.array([])

    # Собираем массивы N_list, typ_i_list (аналог MATLAB)
    N_list = [0]*natom
    typ_i_list = [0]*natom
    idx = 0
    for i_sp in range(species):
        for _ in range(numIons[i_sp]):
            N_list[idx] = numIons[i_sp]
            typ_i_list[idx] = i_sp + 1  # 1-базовая
            idx += 1

    # Подготовка для главных циклов
    # (тот же расчёт, что и в исходном коде, чтобы совпадал lengthX/Y/Z)
    vect = np.zeros((13, 3), dtype=float)
    vect[0]  = lattice[0]
    vect[1]  = lattice[1]
    vect[2]  = lattice[2]
    vect[3]  = vect[0] + vect[1]
    vect[4]  = vect[0] - vect[1]
    vect[5]  = vect[0] + vect[2]
    vect[6]  = vect[0] - vect[2]
    vect[7]  = vect[1] + vect[2]
    vect[8]  = vect[2] - vect[1]
    vect[9]  = vect[0] + vect[1] + vect[2]
    vect[10] = vect[0] + vect[1] - vect[2]
    vect[11] = vect[0] - vect[1] + vect[2]
    vect[12] = -vect[0] + vect[1] + vect[2]

    abs_vect = np.sqrt(np.sum(vect**2, axis=1))
    lengthX = int(np.ceil((RmaxFing + max(abs_vect)) / min(abs_vect))) + 1
    lengthY = lengthX
    lengthZ = lengthX

    signum = np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1, -1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1, -1]
    ], dtype=int)

    condition = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=int)

    # Списки для накопления результатов
    dist_matrix_list = []
    typ_j_list = []

    Nfull = 0  # аналог счётчика заполненных строк

    # Главные циклы: i, j, k, quad, current_cell
    for i in range(lengthX + 1):
        quit_marker_x = 1
        for j in range(lengthY + 1):
            quit_marker_y = 1
            for k in range(lengthZ + 1):
                if dimension == 2 and k > 0:
                    continue
                if dimension == 0 and (i + j + k) > 0:
                    continue

                quit_marker_z = 1

                for quad in range(8):
                    cond_sum = (condition[quad, 0]*(i == 0) +
                                condition[quad, 1]*(j == 0) +
                                condition[quad, 2]*(k == 0))
                    if cond_sum == 0:
                        # Цикл по атомам исходной ячейки (current_cell)
                        for current_cell in range(natom):
                            marker = 0
                            # Вычисляем дробные координаты «образа»
                            fx = coordinates[current_cell, 0] + signum[quad, 0]*i
                            fy = coordinates[current_cell, 1] + signum[quad, 1]*j
                            fz = coordinates[current_cell, 2] + signum[quad, 2]*k

                            # Разница с ВСЕМИ атомами исходной ячейки (в дробных координатах)
                            dx = fx - coordinates[:, 0]
                            dy = fy - coordinates[:, 1]
                            dz = fz - coordinates[:, 2]

                            # Переводим в декартовы координаты
                            frac_shifts = np.column_stack([dx, dy, dz])  # shape=(natom, 3)
                            cart = frac_shifts @ lattice               # shape=(natom, 3)
                            dist_sq = np.sum(cart**2, axis=1)
                            mask = dist_sq < (RmaxFing**2)

                            if np.any(mask):
                                # Нашли атом(ы), значит выходим из quit
                                quit_marker_z = 0
                                quit_marker_y = 0
                                quit_marker_x = 0

                                # Если впервые для этого current_cell (marker=0)
                                # и мы уже "превысили" начальный размер N_list => расширяем
                                if marker == 0 and Nfull >= natom:
                                    N_list.append(N_list[current_cell])

                                Nfull += (1 - marker)
                                marker = 1

                                # Формируем строку расстояний (0 - где вне RmaxFing)
                                distances = np.zeros(natom, dtype=float)
                                distances[mask] = np.sqrt(dist_sq[mask])

                                # Определяем тип
                                typ_j_coef = typ_i_list[current_cell]

                                # Логика "if (i + j + k + (current_cell+1)) == 1" одинаковая
                                dist_matrix_list.append(distances)
                                typ_j_list.append(typ_j_coef)

                if quit_marker_z == 1:
                    break
            if quit_marker_y == 1:
                break
        if quit_marker_x == 1:
            break

    # Преобразуем в numpy-массивы
    N = np.array(N_list, dtype=int)
    typ_i = np.array(typ_i_list, dtype=int)

    if len(dist_matrix_list) > 0:
        dist_matrix = np.vstack(dist_matrix_list)
    else:
        dist_matrix = np.zeros((0, natom), dtype=float)

    typ_j = np.array(typ_j_list, dtype=int)

    return N, V, dist_matrix, typ_i, typ_j
