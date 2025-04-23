import numpy as np

def initialize_lattice(L=50, init_type='random'):
    """
    创建一个 2L x 2L 的自旋格子，值为 +1 或 -1
    init_type 可选：'random'（随机）、'up'（全 +1）、'down'（全 -1）
    """
    N = 2 * L
    if init_type == 'random':
        return np.random.choice([-1, 1], size=(N, N))
    elif init_type == 'up':
        return np.ones((N, N), dtype=int)
    elif init_type == 'down':
        return -np.ones((N, N), dtype=int)
    else:
        raise ValueError("init_type must be 'random', 'up', or 'down'")

def get_neighbors(i, j, N):
    """
    获取点 (i, j) 的4个周期性邻居坐标
    """
    return [
        ((i + 1) % N, j),
        ((i - 1 + N) % N, j),
        (i, (j + 1) % N),
        (i, (j - 1 + N) % N)
    ]

def sum_neighbor_spins(i, j, lattice):
    """
    返回 (i, j) 位置的邻居自旋和
    """
    N = lattice.shape[0]
    neighbors = get_neighbors(i, j, N)
    return sum(lattice[x, y] for x, y in neighbors)

# === 测试代码（放最后） ===
if __name__ == "__main__":
    # 创建一个 6x6 的随机格子
    lattice = initialize_lattice(L=3, init_type='random')
    print("Lattice:")
    print(lattice)

    # 打印 (0, 0) 点的邻居坐标
    print("\nNeighbors of (0, 0):")
    print(get_neighbors(0, 0, lattice.shape[0]))

    # 打印 (0, 0) 点邻居的自旋和
    print("\nSum of neighbor spins at (0, 0):")
    print(sum_neighbor_spins(0, 0, lattice))
