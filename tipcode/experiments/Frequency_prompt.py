import torch

def sliding_window_sum(tensor, window_size):
    # 使用unfold方法进行滑窗操作
    unfolded = tensor.unfold(1, window_size, 1).unfold(2, window_size, 1)
    # 计算每个滑窗的和
    sums = unfolded.sum(dim=(-1, -2))
    return sums

def top_k_windows(sum_tensor, k):
    # 找到和最大的k个滑窗
    k_max_values, k_max_indices = torch.topk(sum_tensor.contiguous().view(-1), k)
    # 转换索引为二维坐标
    y_indices, x_indices = torch.div(k_max_indices, sum_tensor.shape[1], rounding_mode='floor'), k_max_indices % sum_tensor.shape[1]
    return y_indices, x_indices

def get_values_coordinates(tensor, y_indices, x_indices, n, window_size):
    coordinates = {'max': [], 'min': []}
    for y, x in zip(y_indices, x_indices):
        window = tensor[y:y+window_size, x:x+window_size]
        max_values, max_indices = torch.topk(window.contiguous().view(-1), n)
        min_values, min_indices = torch.topk(window.contiguous().view(-1), n, largest=False)
        # 将线性索引转换为2D坐标
        max_coords = [(idx.item() // window_size, idx.item() % window_size) for idx in max_indices]
        min_coords = [(idx.item() // window_size, idx.item() % window_size) for idx in min_indices]
        coordinates['max'].extend([(y + dy, x + dx) for dy, dx in max_coords])
        coordinates['min'].extend([(y + dy, x + dx) for dy, dx in min_coords])
    return coordinates

# 示例参数
C, H, W = 3, 10, 10  # 张量尺寸
g = 3  # 滑窗大小
k = 5  # 前k个滑窗
n = 2  # 每个滑窗内的最大/最小n个值

# 创建两个随机张量
tensor1 = torch.rand(C, H, W)
tensor2 = torch.rand(C, H, W)

# 对每个通道进行操作
def point_prompt(tensor1,tensor2,k=5,g=3,n=2):
    results = []
    for c in range(C):
    # 第一个张量的滑窗和
        sum_tensor = sliding_window_sum(tensor1[c], g)
    # 找到和最大的k个滑窗
        y_indices, x_indices = top_k_windows(sum_tensor, k)
    # 在第二个张量中找到对应的滑窗并获取坐标
        coordinates = get_values_coordinates(tensor2[c], y_indices, x_indices, n, g)
        results.append(coordinates)

    point_final = []
    for c in range(C):
        point_final.append(results[c]['max']+results[c]['min'])
#print(len(point_final[0][0]))
    point_final = torch.tensor(point_final)
    return point_final

res = point_prompt(tensor1,tensor2)
print(res.size())
