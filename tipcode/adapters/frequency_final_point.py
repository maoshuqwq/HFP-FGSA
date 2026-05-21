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

def get_values_coordinates_and_threshold(tensor, y_indices, x_indices, n, window_size):
    coordinates = {'max': [], 'min': []}
    thresholded_values = {'max': [], 'min': []}
    for y, x in zip(y_indices, x_indices):
        window = tensor[ y:y+window_size, x:x+window_size]
        max_values, max_indices = torch.topk(window.contiguous().view(-1), n)
        min_values, min_indices = torch.topk(window.contiguous().view(-1), n, largest=False)

        # 将线性索引转换为2D坐标，并检查值是否大于0.5
        for idx, val in zip(max_indices, max_values):
            dy, dx = idx.item() // window_size, idx.item() % window_size
            coordinates['max'].append((y + dy, x + dx))
            thresholded_values['max'].append(1 if val > 0.5 else 0)

        for idx, val in zip(min_indices, min_values):
            dy, dx = idx.item() // window_size, idx.item() % window_size
            coordinates['min'].append((y + dy, x + dx))
            thresholded_values['min'].append(1 if val > 0.5 else 0)

    return coordinates, thresholded_values

'''
results = []
for c in range(C):
    # 第一个张量的滑窗和
    sum_tensor = sliding_window_sum(tensor1[c], g)
    # 找到和最大的k个滑窗
    y_indices, x_indices = top_k_windows(sum_tensor, k)
    # 在第二个张量中找到对应的滑窗并获取坐标和阈值处理后的值
    coords, vals = get_values_coordinates_and_threshold(tensor2[c], y_indices, x_indices, n, g)
    results.append((coords, vals))
'''
# 对每个通道进行操作
def point_prompt(tensor1,tensor2,k=5,g=32,n=2):
    C,h,w = tensor1.size()
    results = []
    for c in range(C):
    # 第一个张量的滑窗和
        sum_tensor = sliding_window_sum(tensor1[c], g)
    # 找到和最大的k个滑窗
        y_indices, x_indices = top_k_windows(sum_tensor, k)
    # 在第二个张量中找到对应的滑窗并获取坐标
        coordinates,vals = get_values_coordinates_and_threshold(tensor2[c], y_indices, x_indices, n, g)
        results.append((coordinates,vals))

    point_final_positions = []
    point_final_values = []
    for c in range(C):
        point_final_positions.append(results[c][0]['max']+results[c][0]['min'])
        point_final_values.append(results[c][1]['max'] + results[c][1]['min'])
#print(len(point_final[0][0]))
    point_final_positions = torch.tensor(point_final_positions)
    point_final_values = torch.tensor(point_final_values)
    return point_final_positions,point_final_values
# # 输出结果
# for channel, (coords, vals) in enumerate(results):
#     print(f"Channel {channel}:")
#     for key in coords:
#         print(f"{key.capitalize()} value coordinates:")
#         print(" ", coords[key])
#         print(f"{key.capitalize()} thresholded values:")
#         print(" ", vals[key])
