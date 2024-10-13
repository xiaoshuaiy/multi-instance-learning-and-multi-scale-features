import scipy.io as sio
def get_centers(patch_size,shape):
    block_centers = []
    for l in range(0, shape[0], patch_size):
        for w in range(0, shape[1], patch_size):
            for h in range(0, shape[2], patch_size):
                # 计算区域中心点
                center = patch_size // 2
                # print(center)
                center_l = l + center  # 区域在 l 维度的中心点索引
                center_w = w + center  # 区域在 w 维度的中心点索引
                center_h = h + center  # 区域在 h 维度的中心点索引
                center_point = (center_l, center_w, center_h)
                block_centers.append(center_point)
    sio.savemat(
        'Big_block_template_center_size{}_num{}.mat'.format(
           patch_size,len(block_centers)),
        {"patch_centers": block_centers})


if __name__ == '__main__':
    get_centers(30,(120,150,120))