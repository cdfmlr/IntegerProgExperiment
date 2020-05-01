import numpy as np
from scipy import optimize


def hungarian_assignment(cost_matrix):
    """
    hungarian_assignment 指派问题的匈牙利解法

    :param cost_matrix: 指派问题的系数矩阵
    :return: np.where(指派)
    """

    cost_matrix = np.asarray(cost_matrix)
    if len(cost_matrix.shape) != 2:
        raise ValueError("expected a matrix (2-d array), got a %r array"
                         % (cost_matrix.shape,))

    # The algorithm expects more columns than rows in the cost matrix.
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    state = _Hungarian(cost_matrix)
    step = None if 0 in cost_matrix.shape else _simplize4zero
    cnt = 0
    while step:
        step = step(state)
        cnt += 1
        if cnt > 1000:  # 防止意外陷入死循环，把错误的情况交给 optimize.linear_sum_assignment 处理
            print("[ERROR] hungarian_assignment Failed, Try optimize.linear_sum_assignment")
            return optimize.linear_sum_assignment(cost_matrix)

    if transposed:
        assigned = state.assigned.T
    else:
        assigned = state.assigned
    return np.where(assigned == 1)
    # return np.array(state.assigned == True, dtype=int)


class _Hungarian(object):
    """
    State of the Hungarian algorithm.
    """

    def __init__(self, cost_matrix):
        self.cost = np.array(cost_matrix)

        r, c = self.cost.shape
        self.row_covered = np.zeros(r, dtype=bool)
        self.col_covered = np.zeros(c, dtype=bool)
        self.assigned = np.zeros((r, c), dtype=int)


def _simplize4zero(state: _Hungarian):
    """
    step1. 变换指派问题的系数矩阵(c_{ij})为(b_{ij})，使在(bij)的各行各列中都出现0元素，即:

        1. 从(c_{ij})的每行元素都减去该行的最小元素;
        2. 再从所得新系数矩阵的每列元素中减去该列的最小元素。

    :param state: _Hungarian, State of the Hungarian algorithm.
    :return: next step: _try_assign
    """
    # 从(c_{ij})的每行元素都减去该行的最小元素
    state.cost -= state.cost.min(axis=1)[:, np.newaxis]
    # 再从所得新系数矩阵的每列元素中减去该列的最小元素
    state.cost -= state.cost.min(axis=0)[np.newaxis, :]

    return _try_assign


def _try_assign(state: _Hungarian):
    """
    step2. 进行试指派，以寻求最优解。
    在(b_{ij})中找尽可能多的独立0元素，
    若能找出n个独立0元素，就以这n个独立0元素对应解矩阵(x_{ij})中的元素为1，其余为0，这就得到最优解。

    找独立0元素的步骤为:
        1. __assign_row: 从只有一个0元素的行开始，给该行中的0元素加圈，记作◎。
            然后划去◎所在列的其它0元素，记作Ø ;这表示该列所代表的任务已指派完，不必再考虑别人了。
            依次进行到最后一行。
        2. __assign_col: 从只有一个0元素的列开始(画Ø的不计在内)，给该列中的0元素加圈，记作◎;
            然后划去◎所在行的0元素，记作Ø ，表示此人已有任务，不再为其指派其他任务。
            依次进行到最后一列。
        3. __assign_single_zeros: 若仍有没有划圈且未被划掉的0元素，则同行(列)的0元素至少有两个，
            比较这行各0元素所在列中0元素的数目，选择0元素少的这个0元素加圈(表示选择性多的要“礼让”选择性少的)。
            然后划掉同行同列 的其它0元素。
            可反复进行，直到所有0元素都已圈出和划掉为止。
        4. 若◎元素的数目m等于矩阵的阶数n(即 m=n)，那么这指派问题的最优解已得到。
            若 m < n, 则转入下一步。

    :param state: _Hungarian, State of the Hungarian algorithm.
    :return: next step: None if best reached else _draw_lines
    """
    state.assigned = np.zeros(state.cost.shape, dtype=int)

    __assign_row(state)
    __assign_col(state)
    __assign_single_zeros(state)

    assigned_zeros = np.where(state.assigned == 1)[0]
    if len(assigned_zeros) == len(state.cost):
        # 若◎元素的数目m等于矩阵的阶数n(即:m=n)，那么这指派问题的最优解已得到
        return None
    elif len(assigned_zeros) < len(state.cost):
        return _draw_lines

    raise RuntimeError(assigned_zeros)


def __assign_row(state: _Hungarian):
    """
    step2.1. (Called by _try_assign) 从只有一个0元素的行开始，给该行中的0元素加圈，记作◎。
            然后划去◎所在列的其它0元素，记作Ø ;这表示该列所代表的任务已指派完，不必再考虑别人了。
            依次进行到最后一行。

    :param state: _Hungarian, State of the Hungarian algorithm.
    :return: None
    """
    start_flag = True
    for i, row in enumerate(state.cost):  # 从只有一个0元素的行开始，依次进行到最后一行。
        zero_idx = np.where(row == 0)[0]
        if not start_flag or len(zero_idx) == 1:  # 只有一个0元素的行
            start_flag = False
            j = zero_idx[np.random.randint(len(zero_idx))]
            if state.assigned[i, j] == 0:
                for k, _ in enumerate(state.cost.T[j]):
                    if state.cost[k, j] == 0:
                        state.assigned[k, j] = -1  # 划去◎所在列的其它0元素，记作Ø，表示该列所代表的任务已 指派完，不必再考虑别人了
                state.assigned[i, j] = 1  # 给该行中的0元素加圈，记作◎


def __assign_col(state: _Hungarian):
    """
    step2.2. (Called by _try_assign) 从只有一个0元素的列开始(画Ø的不计在内)，给该列中的0元素加圈，记作◎;
            然后划去◎所在行的0元素，记作Ø ，表示此人已有任务，不再为其指派其他任务。
            依次进行到最后一列。

    :param state: _Hungarian, State of the Hungarian algorithm.
    :return: None
    """
    start_flag = True
    for i, col in enumerate(state.cost.T):  # 从只有一个0元素的列开始(画Ø的不计在内), 依次进行到最后一列。
        zero_idx = np.where(col == 0)[0]
        zero_idx_except_slashed = np.where(state.assigned.T[i][zero_idx] == 0)[0]
        # if not start_flag or (state.assigned[zero_idx[0]][i] == 0 and len(zero_idx_except_slashed) == 1):
        if not start_flag or (len(zero_idx_except_slashed) == 1):  # 只有一个0元素的列(画Ø的不计在内)
            start_flag = False
            j = zero_idx[np.random.randint(len(zero_idx))]
            if state.assigned[j, i] == 0:
                for k, _ in enumerate(state.cost[j]):
                    if state.cost[j, k] == 0:
                        state.assigned[j, k] = -1  # 划去◎所在列的其它0元素，记作Ø，表示该列所代表的任务已 指派完，不必再考虑别人了
                state.assigned[j, i] = 1  # 给该行中的0元素加圈，记作◎


def __assign_single_zeros(state: _Hungarian):
    """
    step2.3. (Called by _try_assign) 若仍有没有划圈且未被划掉的0元素，则同行(列)的0元素至少有两个，
            比较这行各0元素所在列中0元素的数目，选择0元素少的这个0元素加圈(表示选择性多的要“礼让”选择性少的)。
            然后划掉同行同列 的其它0元素。
            可反复进行，直到所有0元素都已圈出和划掉为止。

    :param state: _Hungarian, State of the Hungarian algorithm.
    :return: None
    """
    cnt = 0
    while cnt < 100:
        cnt += 1
        zx, zy = np.where(state.cost == 0)  # 0元素
        for i in range(len(zx)):
            if state.assigned[zx[i], zy[i]] == 0:  # 没有划圈且未被划掉的0元素
                zeros_idx_in_row = np.where(state.cost[zx[i]] == 0)[0]  # 这行各0元素
                if len(zeros_idx_in_row) > 1:
                    # 比较这行各0元素所在列中0元素的数目
                    zs_each_col = [(z, len(np.where(state.cost.T[z] == 0)[0])) for z in zeros_idx_in_row]
                    min_zeros_idx = min(zs_each_col, key=lambda x: x[1])[0]
                    # 选择0元素少的这个0元素加圈(表示选择性多的要“礼让”选择性少的)
                    state.assigned[zx[i], zeros_idx_in_row] = -1
                    for k, _ in enumerate(state.cost.T[min_zeros_idx]):
                        if state.cost[k, min_zeros_idx] == 0:
                            state.assigned[k, min_zeros_idx] = -1  # 划去◎所在列的其它0元素，记作Ø，表示该列所代表的任务已 指派完，不必再考虑别人了
                    state.assigned[zx[i], min_zeros_idx] = 1
                    continue
                zeros_idx_in_col = np.where(state.cost.T[zy[i]] == 0)[0]  # 这列各0元素
                if len(zeros_idx_in_col) > 1:
                    # 比较这列各0元素所在行中0元素的数目
                    zs_each_row = [(z, len(np.where(state.cost[z] == 0)[0])) for z in zeros_idx_in_col]
                    min_zeros_idx = min(zs_each_row, key=lambda x: x[1])[0]
                    # 选择0元素少的这个0元素加圈(表示选择性多的要“礼让”选择性少的)
                    state.assigned[zeros_idx_in_col, zx[i]] = -1
                    for k, _ in enumerate(state.cost[min_zeros_idx]):
                        if state.cost[min_zeros_idx, k] == 0 and state.assigned[min_zeros_idx, k] == 0:
                            state.assigned[min_zeros_idx, k] = -1  # 划去◎所在列的其它0元素，记作Ø，表示该列所代表的任务已 指派完，不必再考虑别人了
                    state.assigned[min_zeros_idx, zy[i]] = 1
        zx, zy = np.where(state.cost == 0)  # 0元素
        if not any([state.assigned[zx[i], zy[i]] == 0 for i in range(len(zx))]):  # 所有0元素都已圈出和划掉
            return

    raise RuntimeError("Too many iters:", state.assigned)


def _draw_lines(state: _Hungarian):
    """
    step3. 用最少的直线通过所有0元素。具体方法为:

        1. 对没有◎的行打“√”;
        2. 对已打“√” 的行中所有含Ø元素的列打“√” ;
        3. 再对打有“√”的列中含◎ 元素的行打“√” ;
        4. 重复2、 3直到得不出新的打√号的行、列为止;
        5. 对没有打√号的行画横线，有打√号的列画纵线，这就得到覆 盖所有0元素的最少直线数 l 。

    注: l 应等于 m， 若不相等，说明试指派过程有误，回到第2步，另行试指派;
        若 l = m < n，表示还不能确定最优指派方案，须再变换当前的系数矩阵，以找到n个独立的0元素，为此转第4步。

    :param state: _Hungarian, State of the Hungarian algorithm.
    :return: _transform_cost if assignment is correct else _try_assign
    """
    state.row_covered[:] = 0
    state.col_covered[:] = 0
    # 1、对没有◎的行打“√”;
    for i, row in enumerate(state.assigned):
        assigned_zeros = np.where(row == 1)[0]
        if len(assigned_zeros) == 0:
            state.row_covered[i] = True

    old_row_covered = np.zeros(state.row_covered.shape)
    old_col_covered = np.zeros(state.row_covered.shape)

    while np.any(state.row_covered != old_row_covered) or np.any(state.col_covered != old_col_covered):
        # 2、对已打“√” 的行中所有含Ø元素的列打“√”
        for i, covered in enumerate(state.row_covered):
            if covered:
                slashed_zeros = np.where(state.assigned[i, :] == -1)[0]
                state.col_covered[slashed_zeros] = True

        # 3、再对打有“√”的列中含◎元素的行打“√”
        for i, covered in enumerate(state.col_covered):
            if covered:
                assigned_zeros = np.where(state.assigned[:, i] == 1)[0]
                state.row_covered[assigned_zeros] = True
        # 重复2、3直到得不出新的打√号的行、列为止;
        old_row_covered = state.row_covered.copy()
        old_col_covered = state.col_covered.copy()

    # 对没有打√号的行画横线，有打√号的列画纵线
    state.row_covered = (state.row_covered == False)
    # ls: 覆盖所有0元素的最少直线数
    ls = len(np.where(state.row_covered == True)[0]) + len(np.where(state.col_covered == True)[0])
    assigned_zeros = np.where(state.assigned == 1)[0]
    if ls == len(assigned_zeros) and ls < len(state.cost):
        return _transform_cost
    elif ls == len(assigned_zeros) and ls == len(state.cost):
        return None
    # 不相等，说明试指派过程有误，回到第2步，另行试指派;
    return _try_assign
    # raise RuntimeError(ls, len(assigned_zeros), len(state.cost))


def _transform_cost(state: _Hungarian):
    """
    step4. 变换矩阵(b_{ij})以增加0元素

    在没有被直线通过的所有元素中找出最小值，
    没有被直线通过的所有元素减去这个最小元素;
    直线交点处的元素加上这个最小值。
    新系数矩阵的最优解和原问题仍相同。
    转回第2步。

    :param state: _Hungarian, State of the Hungarian algorithm.
    :return: _try_assign
    """
    # 找出被直线通过的所有元素
    row_idx_covered = np.where(state.row_covered == True)[0]
    col_idx_covered = np.where(state.col_covered == True)[0]
    # 找出没有被直线通过的所有元素
    row_idx_not_covered = np.where(state.row_covered == False)[0]
    col_idx_not_covered = np.where(state.col_covered == False)[0]

    # 在没有被直线通过的所有元素中找出最小值
    min_element = state.cost[row_idx_not_covered].T[col_idx_not_covered].min()

    # 没有被直线通过的所有元素减去这个最小元素
    for r in row_idx_not_covered:
        for c, _ in enumerate(state.cost[r]):
            if c in col_idx_not_covered:
                state.cost[r, c] -= min_element
    # state.cost[row_idx_not_covered].T[col_idx_not_covered].T -= min_element

    # 直线交点处的元素加上这个最小值
    # state.cost[row_idx_covered].T[col_idx_covered].T += min_element
    for r in row_idx_covered:
        for c, _ in enumerate(state.cost[r]):
            if c in col_idx_covered:
                state.cost[r][c] += min_element

    return _try_assign


def _test_simplize4zero():
    """
    expected: [[ 0 13  7  0] [ 6  0  6  9] [ 0  5  3  2] [ 0  1  0  0]]
    """
    c = [[2, 15, 13, 4], [10, 4, 14, 15], [9, 14, 16, 13], [7, 8, 11, 9]]
    print("_test_simplize4zero:")
    s = _Hungarian(c)
    _simplize4zero(s)
    print(s.cost)


def _test_try_assign():
    """
    expected: [[-1  0  0  1] [ 0  1  0  0] [ 1  0  0  0] [-1  0  1 -1]]
    """
    b = [[0, 13, 7, 0], [6, 0, 6, 9], [0, 5, 3, 2], [0, 1, 0, 0]]
    s = _Hungarian(b)
    print("_test_try_assign:")
    _try_assign(s)
    print(s.assigned)


def _test_draw_lines_transform_cost():
    """
    expected: row: [ True  True False  True False]
              col: [ True False False False False]
              transformed cost: [[ 7  0  2  0  2] [ 4  3  0  0  0] [ 0  8  3  5  0] [11  8  0  0  4] [ 0  4  1  4  3]]
    """
    print("_test_draw_lines:")
    c = [[12, 7, 9, 7, 9], [8, 9, 6, 6, 6], [7, 17, 12, 14, 9], [15, 14, 6, 6, 10], [4, 10, 7, 10, 9]]
    s = _Hungarian(c)
    _simplize4zero(s)
    _try_assign(s)
    _draw_lines(s)
    print("row:", s.row_covered)
    print("col:", s.col_covered)
    _transform_cost(s)
    print("transformed cost:\n", s.cost)


def _test1():
    print("Test1\n" + '-' * 10)
    c = [[2, 15, 13, 4], [10, 4, 14, 15], [9, 14, 16, 13], [7, 8, 11, 9]]
    r = hungarian_assignment(c)
    print(r)
    assert np.all(r == np.array([[0, 1, 2, 3], [3, 1, 0, 2]]))
    m = np.zeros(np.array(c).shape, dtype=int)
    m[r] = 1
    print(m)


def _test2():
    # from scipy.optimize import linear_sum_assignment
    print("\nTest2\n" + '-' * 10)
    c = [[12, 7, 9, 7, 9], [8, 9, 6, 6, 6], [7, 17, 12, 14, 9], [15, 14, 6, 6, 10], [4, 10, 7, 10, 9]]
    r = hungarian_assignment(c)
    print(r)
    assert np.all(r == np.array([[0, 1, 2, 3, 4], [1, 3, 4, 2, 0]])) or np.all(
        r == np.array([[0, 1, 2, 3, 4], [1, 2, 4, 3, 0]]))
    # 两个答案，一个是书上的，一个是 scipy.optimize.linear_sum_assignment 解出来的，总消耗是一样的。
    m = np.zeros(np.array(c).shape, dtype=int)
    m[r] = 1
    print(m)


def _test3():
    print("\nTest3\n" + '-' * 10)
    c = [[6, 7, 11, 2], [4, 5, 9, 8], [3, 1, 10, 4], [5, 9, 8, 2]]
    r = hungarian_assignment(c)
    print(r)
    m = np.zeros(np.array(c).shape, dtype=int)
    m[r] = 1
    print(m)


def _test4():
    print("\nTest4\n" + '-' * 10)
    c = [[7, 5, 9, 8, 11], [9, 12, 7, 11, 9], [8, 5, 4, 6, 9], [7, 3, 6, 9, 6], [4, 6, 7, 5, 11]]
    r = hungarian_assignment(c)
    print(r)
    m = np.zeros(np.array(c).shape, dtype=int)
    m[r] = 1
    print(m)


if __name__ == "__main__":
    # _test_simplize4zero()             # pass
    # _test_try_assign()                # pass
    # _test_draw_lines_transform_cost() # pass
    _test1()
    _test2()
    _test3()
    _test4()
