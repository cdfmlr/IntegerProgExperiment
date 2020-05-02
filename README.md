# IntegerProgExperiment
 线性规划课程实验——整数规划问题的求解方法

## 仓库内容

```
.
├── BranchAndBound		整数线性规划问题的分支定界法实现
├── CuttingPlane		整数线性规划问题的割平面法实现
├── HungarianAssignment	指派问题的匈牙利算法实现
├── MonteCarlo			整数规划问题的蒙特卡洛法实现
└── experiment.py		各算法实现的使用示例
```

See Docs Here：https://cdfmlr.github.io/IntegerProgExperiment/

## 实现简介

### 分支定界法

> `BranchAndBound`: 整数线性规划问题的「分支定界法」实现。

#### 问题模型

```
Minimize:     c^T * x

Subject to:   A_ub * x <= b_ub
              A_eq * x == b_eq
              (x are integers)
```

#### 算法思路

1. 求整数规划松弛问题的最优解， 若松弛问题的最优解满足整数要求，则得到整数规划的最优解; 否则转下一步;

2. 分枝、定界与剪枝

   选一个非整数解的变量xi，在松弛问题中加上约束:
   $x_i≤[x_i] 和 x_i≥[x_i]+1$($[x_i]$: 小于 $x_i$ 的最大整数) 
   
   构成两个新的松弛问题，称为分枝。
   
   检查所有分枝的最优解及最优值，进行定界、剪枝: 
   
   - 若某分枝的最优解是整数并且目标函数值大于其它分枝的最优值 ，则将其它分枝剪去不再计算; 
   - 若还存在非整数解并且最优值大于整数解的最优值，转 2)，需要继续分枝、定界、剪枝，直到得到最优解 $z^*$。

#### 实现 API

```python
BranchAndBound.branch_and_bound(c, A_ub, b_ub, A_eq, b_eq, bounds, bnbTreeNode=None)
```

该算法实现使用递归调用实现，底层对于松弛问题求解调用 `scipy.optimize.linprog` 完成。

参数 `c, A_ub, b_ub, A_eq, b_eq, bounds` 的要求与 `scipy.optimize.linprog` 的同名参数完全相同。

在使用过程中，可以提供一个 `BranchAndBound.BnBTreeNode` 实例作为根节点来记录求解过程，得到一个求解过程的树形图。

#### 使用示例

对于问题：

```
min 3 * x_0 + 4 * x_1 + x_2

s.t. x_0 + 6 * x_1 + 2 * x_2 >= 5
	 2 * x_0 >= 3
	 x_0, x_1, x_2 >= 0, 为整数
```

编写如下代码使用实现的接口进行求解：

```python
import BranchAndBound

c = [3, 4, 1]
A_ub = [[-1, -6, -2], [-2, 0, 0]]
b_ub = [-5, -3]
bounds = [(0, None), (0, None), (0, None)]
tree = BranchAndBound.BnBTree()
r = BranchAndBound.branch_and_bound(c, A_ub, b_ub, None, None, bounds, tree.root)
print(r)
print(tree)
```

输出：

```
{'success': True, 'x': array([2., 0., 2.]), 'fun': 8.0}
-- Root: [1.5  0.   1.75] -> 6.250000000000001
	|-- x[0] <= 1: nan -> 1.0
	|-- x[0] >= 2: [2.  0.  1.5] -> 7.5
	|	|-- x[2] <= 1: [2.  0.16666667 1.] -> 7.666666666666667
	|	|	|-- x[1] <= 0: [3. 0. 1.] -> 10.0
	|	|	|-- x[1] >= 1: [2. 1. 0.] -> 10.0
	|	|-- x[2] >= 2: [2. 0. 2.] -> 8.0
```

### 割平面法

> `CuttingPlane`: 整数线性规划问题的割平面法实现。

（这个实现有 Bug，我没改，对于部分问题会求解失败，不推荐使用）

#### 问题模型

```
Maximize:     c^T * x

Subject to:   A * x == b
			  (x are integers)
```

#### 算法思路

先不考虑变量的整数约束，求解相应的线性规划， 然后不断增加线性约束条件(即割平面)， 将原可行域割掉不含整数可行解的一部分， 最终得到一个具有整数坐标顶点的可行域， 而该顶点恰好是原整数规划问题的最优解。

#### 实现 API

```python
CuttingPlane.cutting_plane(c, A, b)
```

该算法实现使用递归调用实现，底层对于松弛问题求解调用  [`cdfmlr/SimplexLinprog`](https://github.com/cdfmlr/SimplexLinprog) 中的 `linprog.simplex_method.SimplexMethod` 完成。

参数 `c, A, b` 的要求和 `linprog.simplex_method.SimplexMethod` 完全相同。

使用过程中要注意将问题化为标准型。

#### 使用示例

对于问题：

```
max x_0 + x_1

s.t. 2 * x_0 + x_1 <= 6
	 4 * x_0 + 5 * x_1 <= 20
	 x_0, x_1 >= 0, 为整数
```

编写如下代码使用实现的接口进行求解：

```python
import CuttingPlane

c = [1, 1, 0, 0]
a = [[2, 1, 1, 0], [4, 5, 0, 1]]
b = [6, 20]
r = CuttingPlane.cutting_plane(c, a, b)
print(r)
```

输出：

```python
{'success': True, 'x': array([2., 2., 0., 2., 0.]), 'fun': -0.33333333333333326}
```

### 蒙特卡洛法

> `MonteCarl`: 整数规划问题的蒙特卡洛法实现。

#### 问题模型

```
Minimize:   fun(x)

Subject to: cons(x) <= 0
			(x are integers)
```

#### 算法思路

当所求解问题是某种随机事件出现的概率，或者是某个随机变量的期望值时，通过某种“实验”的方法，以这种事件出现的频率估计这一随机事件的概率，或者得到这个随机变量的某些数字特征，并将其作为问题的解。

对于线性、非线性的整数规划，在一定计算量下可以用这种思想，通过大量的随机试验，得到一个满意解。

#### 实现 API

```python
MonteCarl.monte_carlo(x_nums, fun, cons, bounds, random_times=10**5)
```

Monte carlo 方法可以求解非线性的整数规划，目标函数、约束条件不在由向量、矩阵给出，而可以直接使用两个方法 `fun`、`cons` 来表达。

注意：蒙特卡洛法只能在一定次数的模拟中求一个满意解（通常不是最优的），而且对于每个变量必须给出**有明确上下界的取值范围**。

#### 使用示例

对于问题：

```
max x_0 + x_1

s.t. 2 * x_0 + x_1 <= 6
	 4 * x_0 + 5 * x_1 <= 20
	 0 <= x_0, x_1 <= 100, 为整数
```

编写如下代码使用实现的接口进行求解：

```python
import MonteCarlo

def fun(x):
    return x[0] + x[1]

def cons(x):
    return [
        2 * x[0] + x[1] - 6,
        4 * x[0] + 5 * x[1] - 20,
    ]

bounds = [(0, 100), (0, 100)]
r = MonteCarlo.monte_carlo(2, fun, cons, bounds)
print(r)
```

输出：

```
{'fun': 4, 'x': [1, 3]}
```

### 匈牙利算法

> `HungarianAssignment`: 指派问题的匈牙利算法实现。

#### 问题模型

设 n 个人被分配去做 n 件工作，规定每个人只做一件工作，每 件工作只有一个人去做。已知第i个人去做第j 件工作的效率 ( 时间或费用)为`c_{ij}` (i=1,2,...,n; j=1,2,...,n)并假设 `c_{ij} ≥ 0`。问 应如何分配才能使总效率( 时间或费用)最高?

设决策变量：

```
x_{ij} = 1  若指派第i个人做第j件工作
	  or 0  不指派第i个人做第j件工作
for i, j = 1, 2, ..., n
```

则问题可表示为：

```
\min \sum_i \sum_j C_{i,j} X_{i,j}

s.t. each row is assignment to at most one column, and each column to at most one row.
```

#### 算法思路

step1. 变换指派问题的系数矩阵(`c_{ij}`)为(`b_{ij}`)，使在(`b_{ij}`)的各行各列中都出现0元素，即:

1. 从(`c_{ij}`)的每行元素都减去该行的最小元素;
2. 再从所得新系数矩阵的每列元素中减去该列的最小元素。

step2. 进行试指派，以寻求最优解。

在(`b_{ij}`)中找尽可能多的独立0元素，若能找出n个独立0元素，就以这n个独立0元素对应解矩阵(`x_{ij}`)中的元素为1，其余为0，这就得到最优解。

找独立0元素的步骤为:
1. 从只有一个0元素的行开始，给该行中的0元素加圈，记作◎。然后划去◎所在列的其它0元素，记作Ø ;这表示该列所代表的任务已指派完，不必再考虑别人了。依次进行到最后一行。
2. 从只有一个0元素的列开始(画Ø的不计在内)，给该列中的0元素加圈，记作◎;然后划去◎所在行的0元素，记作Ø ，表示此人已有任务，不再为其指派其他任务。依次进行到最后一列。
3. 若仍有没有划圈且未被划掉的0元素，则同行(列)的0元素至少有两个，比较这行各0元素所在列中0元素的数目，选择0元素少的这个0元素加圈(表示选择性多的要“礼让”选择性少的)。然后划掉同行同列 的其它0元素。可反复进行，直到所有0元素都已圈出和划掉为止。
4. 若◎元素的数目m等于矩阵的阶数n(即 m=n)，那么这指派问题的最优解已得到。若 m < n, 则转入下一步。

step3. 用最少的直线通过所有0元素。具体方法为:

1. 对没有◎的行打“√”;
2. 对已打“√” 的行中所有含Ø元素的列打“√” ;
3. 再对打有“√”的列中含◎ 元素的行打“√” ;
4. 重复2、 3直到得不出新的打√号的行、列为止;
5. 对没有打√号的行画横线，有打√号的列画纵线，这就得到覆 盖所有0元素的最少直线数 l 。

注: l 应等于 m， 若不相等，说明试指派过程有误，回到第2步，另行试指派;
若 l = m < n，表示还不能确定最优指派方案，须再变换当前的系数矩阵，以找到n个独立的0元素，为此转第4步。

step4. 变换矩阵(`b_{ij}`)以增加0元素

在没有被直线通过的所有元素中找出最小值，没有被直线通过的所有元素减去这个最小元素;直线交点处的元素加上这个最小值。新系数矩阵的最优解和原问题仍相同。转回第2步。

#### 实现 API

```python
HungarianAssignment.hungarian_assignment(cost_matrix)
```

调用该函数，会对给定的指派问题试使用上述思路的实现求解。由于该函数设计与 `scipy.optimize.linear_sum_assignment` 基本一致，故在出现问题时，会调用 `linear_sum_assignment` 尝试进一步求解。

#### 使用示例

对于问题：

有一份中文说明书，需译成英、日、德、俄四种文字。分别记作E、J、G、R。现有甲、乙、丙、丁四人。他们将中文说明书翻译成不同语种的说明书所需时间如下表所示。问应指派何人去完成何工作，使所需总时间最少？

| 人员 \ 任务 | E    | J    | G    | R    |
| ---------- | ---- | ---- | ---- | ---- |
| 甲          | 2    | 15   | 13 | 4 |
| 乙          | 10   | 4 | 14 | 15 |
| 丙          | 9    | 14 | 16 | 13 |
| 丁          | 7    | 8 | 11 | 9 |

编写如下代码使用实现的接口进行求解：

```python
import HungarianAssignment

c = [[2, 15, 13, 4], [10, 4, 14, 15], [9, 14, 16, 13], [7, 8, 11, 9]]
r = HungarianAssignment.hungarian_assignment(c)
print(r)
m = np.zeros(np.array(c).shape, dtype=int)
m[r] = 1
print(m)
```

输出：

```
(array([0, 1, 2, 3]), array([3, 1, 0, 2]))
[[0 0 0 1]
 [0 1 0 0]
 [1 0 0 0]
 [0 0 1 0]]
```

## 开放源代码

MIT License

Copyright (c) 2020 cdfmlr