import math
import numpy as np

def approx_erf(datapoints, erf_table):
    """
    Аналог MATLAB-функции approx_erf.m:
    Определение значений erf(...) через таблицу erf_table
    для аргументов от -4.01 до 4.01 с шагом 0.01.

    datapoints : 1D numpy array
    erf_table  : 1D numpy array (803 элементов)

    Возвращает my_erf (массив того же размера, что datapoints).
    """
    my_erf = np.zeros_like(datapoints, dtype=float)
    for i in range(len(datapoints)):
        if datapoints[i] >= 4.0:
            my_erf[i] = 1.0
        elif datapoints[i] <= -4.0:
            my_erf[i] = -1.0
        else:
            # Индекс для таблицы
            x1 = int(math.floor(datapoints[i]/0.01 + 402.0))
            # Берем 4 соседних значения из таблицы
            y3 = erf_table[x1 + 2]
            y2 = erf_table[x1 + 1]
            y1 = erf_table[x1]
            y0 = erf_table[x1 - 1]
            # Полиномиальная интерполяция
            a0 = -y0/6.0 + y1/2.0 - y2/2.0 + y3/6.0
            a1 =  y0/2.0 - y1     + y2/2.0
            a2 = -y0/3.0 - y1/2.0 + y2     - y3/6.0
            a3 =  y1
            q  = (datapoints[i]/0.01 + 402.0 - x1)
            my_erf[i] = ((a0*q + a1)*q + a2)*q + a3
    return my_erf


def fingerprint_calc(Ni, V, dist_matrix, typ_i, typ_j, numIons,
                                Rmax, sigm, delta, dimension):
    """
    Оптимизированная версия fingerprint_calc7, в которой векторизован
    внутренний цикл по j_full (атомам суперячейки).

    Параметры и возвращаемые значения те же, что и в оригинале fingerprint_calc7.
    """
    species = len(numIons)
    Ncell = sum(numIons)
    Nfull = dist_matrix.shape[0]

    normaliser = 1
    if dimension == 0:
        V = 1
        normaliser = 0

    if Ncell == 0:
        # Пустой случай
        return np.array([]), np.array([]), np.array([])

    # Создаём таблицу erf (803 точки)
    erf_table = np.zeros(803, dtype=float)
    for i in range(803):
        erf_table[i] = math.erf((i - 402) / 100.0)

    numBins = int(round(Rmax / delta))

    fing = np.zeros((species * species, numBins), dtype=float)
    if dimension != 0:
        fing[:, 0] = -1.0

    order = np.zeros(Ncell, dtype=float)
    atom_fing = np.zeros((Ncell, species, numBins), dtype=float)

    # Sigma пересчитываем, как в MATLAB-версии
    sigm = sigm / math.sqrt(2.0 * math.log(2.0))

    # Заранее создадим массив "весов" для вычисления order
    # weight_j = numIons[j_sp]/Ncell
    weights = np.array(numIons, dtype=float) / float(Ncell)

    # Цикл по бинам (bins=2..numBins включительно)
    for bin_i in range(2, numBins + 1):

        # Перебираем атомы ячейки (i_cell)
        for i_cell in range(Ncell):
            R0_all = dist_matrix[:, i_cell]  # Все расстояния до i_cell
            ti = typ_i[i_cell] - 1  # тип атома i_cell (0-based)

            # Условие (R0>0) and (|R0 - delta*(bins - 0.5)| < 4*sigma)
            mid = delta*(bin_i - 0.5)
            mask = (R0_all > 0) & (np.abs(R0_all - mid) < 4.0 * sigm)
            if not np.any(mask):
                # Если нет подходящих, ничего не добавляем
                pass
            else:
                R0_masked = R0_all[mask]
                Ni_masked = Ni[mask]  # Ni[j_full]
                tj_masked = typ_j[mask] - 1  # 0-based

                # Считаем diff_upper, diff_lower
                diff_upper = delta*bin_i - R0_masked
                diff_lower = delta*(bin_i - 1) - R0_masked

                # Готовим intervals_1 и intervals_0 (аналог interval[2] и interval[1] в старом коде)
                intervals_1 = 5.0 * np.sign(diff_upper)
                big_mask_1 = (np.abs(diff_upper / math.sqrt(2.0)) <= 5.0 * sigm)
                intervals_1[big_mask_1] = diff_upper[big_mask_1] / (math.sqrt(2.0) * sigm)

                intervals_0 = 5.0 * np.sign(diff_lower)
                big_mask_0 = (np.abs(diff_lower / math.sqrt(2.0)) <= 5.0 * sigm)
                intervals_0[big_mask_0] = diff_lower[big_mask_0] / (math.sqrt(2.0) * sigm)

                # Вызываем approx_erf для этих массивов
                erf0 = approx_erf(intervals_0, erf_table)
                erf1 = approx_erf(intervals_1, erf_table)

                # delt_j = 0.5 * (erf1 - erf0)
                delt_j = 0.5 * (erf1 - erf0)

                # Заполняем atom_fing[i_cell, tj_masked, bin_i-1] += delt_j / (Ni_masked * R0_masked^2)
                # Для этой операции используем np.add.at, чтобы учесть повторяющиеся типы tj_masked
                updates_atom = delt_j / (Ni_masked * (R0_masked**2))
                np.add.at(atom_fing[i_cell, :, bin_i - 1], tj_masked, updates_atom)

                # Аналогично fing[row_index, bin_i-1] += delt_j / (R0_masked^2)
                # row_index = ti * species + tj_masked
                updates_fing = delt_j / (R0_masked**2)
                row_index_masked = ti * species + tj_masked
                np.add.at(fing[:, bin_i - 1], row_index_masked, updates_fing)

            # После обработки всех j_full для фиксированного i_cell, пересчитываем atom_fing
            atom_fing[i_cell, :, bin_i - 1] = (V * atom_fing[i_cell, :, bin_i - 1])/(4.0 * math.pi * delta) - normaliser

            # Обновляем order[i_cell]
            # order(i_cell) += sum_j [ weight_j * delta * (atom_fing[i_cell, j, bin_i-1]^2 ) / ( (V/Ncell)^(1/3) ) ]
            val_sq = atom_fing[i_cell, :, bin_i - 1]**2
            denom = (V / float(Ncell)) ** (1.0/3.0)
            order[i_cell] += delta / denom * np.sum(weights * val_sq)

        # После прохода по всем i_cell, обновляем fing(...) аналогично:
        # fing((i-1)*species + j, bin_i-1) = [ V*fing(...) / (4*pi*numIons(i)*numIons(j)*delta ) ] - normaliser
        # Удобно reshаpe-нуть fing в (species, species, numBins), чтобы избавиться от двойного цикла:
        fing_resh = fing.reshape(species, species, numBins)

        # Для каждого i_sp,j_sp, если numIons[i_sp]*numIons[j_sp]>0, делаем соответствующее деление
        for i_sp in range(species):
            for j_sp in range(species):
                if numIons[i_sp] * numIons[j_sp] > 0:
                    val = fing_resh[i_sp, j_sp, bin_i - 1]
                    val = (V * val) / (4.0 * math.pi * numIons[i_sp] * numIons[j_sp] * delta) - normaliser
                    fing_resh[i_sp, j_sp, bin_i - 1] = val

        # Возвращаем fing назад в исходный shape
        fing = fing_resh.reshape(species*species, numBins)

    # Заключительный шаг: order = sqrt(order)
    order = np.sqrt(order)

    return order, fing, atom_fing
