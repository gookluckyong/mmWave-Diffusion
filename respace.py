import numpy as np
import torch as th

from gaussian_diffusion import GaussianDiffusion

def space_timesteps(num_timesteps, sample_timesteps):
    """
    从原始的扩散步数中，均匀地选择出一个子集作为采样步数。这是实现“跳步”的第一步。

    :param num_timesteps: 原始的总扩散步数 (例如 1000)
    :param section_counts: 采样时希望使用的步数 (例如 50)
    :return: 一个包含采样时所用步数索引的集合 (例如 {0, 20, 40, ..., 980})
    """
    # 计算步长，并按步长均匀采样
    all_steps = [int((num_timesteps/sample_timesteps) * x) for x in range(sample_timesteps)]
    return set(all_steps)

class SpacedDiffusion(GaussianDiffusion):
    """
    一个可以“跳过”基础扩散过程中某些步骤的扩散过程包装器(Wrapper)
    :param use_timesteps: 一个集合，包含了从原始扩散过程中保留下来的时间步索引
    :param kwargs: 用于创建基础 GaussianDiffusion 实例的参数
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["sqrt_etas"])
        
        # 1. 创建一个临时的、包含所有原始步数的 base_diffusion 实例，用于访问原始的 eta 表
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa

        # 2. 根据 use_timesteps 筛选出新的、更短的 eta 表
        new_sqrt_etas = []
        for ii, etas_current in enumerate(base_diffusion.sqrt_etas):
            if ii in self.use_timesteps:
                new_sqrt_etas.append(etas_current)
                self.timestep_map.append(ii)
        # 3. 更新参数，用新的、更短的 eta 表替换旧的  
        kwargs["sqrt_etas"] = np.array(new_sqrt_etas)

        # 4. 调用父类（GaussianDiffusion）的构造函数，但传入的是已经“缩短”了的参数  #    这样，SpacedDiffusion 实例本身就是一个步数更少的扩散过程
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        # 在计算均值和方差时，使用包装后的模型
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        # 在计算训练损失时，同样使用包装后的模型
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model): #将原始的U-Net模型包装起来，使其能够理解被缩短了的时间步。
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.original_num_steps)

class _WrappedModel: #一个模型包装器，负责将“缩短后”的时间步映射回“原始”的时间步
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map # 例如 [0, 20, 40, ..., 980]
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        """
        当 SpacedDiffusion 调用模型时，实际上是调用这里。
        :param ts: “缩短后”的时间步张量 (例如，值在 0-49 之间)。
        """
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype) # 将缩短后的时间步 ts (如 0, 1, 2) 映射回原始的时间步 (如 0, 20, 40)
        new_ts = map_tensor[ts]
        return self.model(x, new_ts, **kwargs) # 将映射后的、原始的时间步 new_ts 传递给真正的U-Net模型  这是因为U-Net是在0-999的完整步数上训练的，它需要原始的时间步信息才能正确工作

