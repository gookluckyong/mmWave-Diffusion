import gaussian_diffusion as gd
from respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(
    *,                           
    schedule_name,
    steps=1000,
    loss_type='mse',
    min_noise_level=0.01,
    kappa=1,
    etas_end=0.99,
    schedule_kwargs=None,
    timestep_respacing=None,
):
    """
    :param schedule_name: eta调度器的名称 (例如 'exponential')
    :param steps: 总的扩散步数 (训练时)
    :param loss_type: 损失类型， 'mse' 或 'weighted_mse'
    :param min_noise_level: 最小噪声水平,用于计算eta调度
    :param kappa: 控制扩散核方差的标量
    :param etas_end: eta调度的结束值
    :param schedule_kwargs: 传递给eta调度器的额外参数 (例如 'power')
    :param timestep_respacing: 采样时要使用的步数。如果为None,则等于 `steps` 例如,训练用1000步,采样用50步,这里就传50
    
    """
    # 1. 根据名称和参数生成噪声调度表 (eta schedule)
    sqrt_etas = gd.get_named_eta_schedule(  #调用 get_named_eta_schedule 返回一个长度为 steps 的数组（或张量） sqrt_etas，每个元素近似似 𝜂_t开根号
            schedule_name=schedule_name,
            num_diffusion_timesteps=steps,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=kappa,
            kwargs=schedule_kwargs,
            )
    
    # 2. 确定损失类型
    if loss_type == 'mse':
        loss_type_enum = gd.LossType.MSE
    elif loss_type == 'weighted_mse':
        loss_type_enum = gd.LossType.WEIGHTED_MSE
    else:
        raise ValueError(f"未知的损失类型: {loss_type}")
    
    # 3. 设置步数跳跃逻辑
    if timestep_respacing is None: #时间步重采样参数处理 若不指定 respacing，则使用全长（即 1000）时间步
        timestep_respacing = steps
    else:
        assert isinstance(timestep_respacing, int)


    # 4. 创建并返回 SpacedDiffusion 实例
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        sqrt_etas=sqrt_etas,
        kappa=kappa,
        loss_type=loss_type_enum
    )


