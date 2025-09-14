import enum
import math

import torch
import numpy as np
import torch as th
import torch.nn.functional as F

from basic_ops import mean_flat


def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    """
    Get a pre-defined eta schedule for the given name.

    The eta schedule library consists of eta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    if schedule_name == 'exponential':
        # ponential = kwargs.get('ponential', None)
        # start = math.exp(math.log(min_noise_level / kappa) / ponential)
        # end = math.exp(math.log(etas_end) / (2*ponential))
        # xx = np.linspace(start, end, num_diffusion_timesteps, endpoint=True, dtype=np.float64)
        # sqrt_etas = xx**ponential
        power = kwargs.get('power', None)
        # etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start)) #增长因子 increaser  设希望 等比 增长（不考虑 power）时，从 etas_start 到 etas_end
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1) #对归一化时间做幂次变换，再乘以(T-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start # “幂次形指数插值”：当 power 调整时，决定指数插值的“弯曲”
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas #Numpy 数组，供上层再喂到 torch（通常在 GaussianDiffusion.__init__ 内转 tensor）

# class ModelMeanType(enum.Enum):
#     """
#     Which type of output the model predicts.
#     """
#     START_X = enum.auto()  # the model predicts x_0

class LossType(enum.Enum):
    MSE = enum.auto()           # simplied MSE
    WEIGHTED_MSE = enum.auto()  # weighted mse derived from KL


def _extract_into_tensor(arr, timesteps, broadcast_shape): #调度参数按批次索引并广播
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None] #while len(res.shape) < len(broadcast_shape): res = res[..., None]：逐次增加尾部维度，直到维度数量和目标一致；此时形状如 [B,1,1,...]。
    return res.expand(broadcast_shape)

class GaussianDiffusion:
    """

    :param sqrt_etas: 每个扩散时间步的eta值的一维numpy数组.
    :param kappa: 控制扩散核方差的标量
    :param model_mean_type: 一个ModelMeanType,决定模型输出什么.
    :param loss_type:一个LossType,决定使用的损失函数.
    """

    def __init__(
        self,
        *,
        sqrt_etas,
        kappa,
        loss_type,
    ):
        self.kappa = kappa
        self.loss_type = loss_type

        # Use float64 for accuracy.
        self.sqrt_etas = sqrt_etas
        self.etas = sqrt_etas**2
        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0) 计算后验 q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
                self.posterior_variance[1], self.posterior_variance[1:]
                ) #方差在 t=0（首步）会为 0，于是代码构造 posterior_variance_clipped 把第 0 个 logVar 用第 1 个代替避免 log(0)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas #对 x_t 的系数
        self.posterior_mean_coef2 = self.alpha / self.etas #对 x_0 的系数

        # weight for the mse loss 加权 MSE 权重 (weight_loss_mse) 源自把一步 KL 展开后对不同参数化（预测 x0 / residual / ε）MSE 的系数归一化  计算MSE损失的权重

        weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas)**2


        # self.weight_loss_mse = np.append(weight_loss_mse[1],  weight_loss_mse[1:])
        self.weight_loss_mse = weight_loss_mse


    def q_sample(self, x_start, y, t, noise=None): #前向传播 实现采样公式  x_t=x_0+η_t(y-x_0)+ksqrt(η_t)ε
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).
        对数据进行给定步数的扩散，即从 q(x_t | x_0, y) 中采样
        :param x_start:初始纯净信号 (纯净呼吸), [N x C x ...].
        :param y: 条件信号 (含噪雷达相位), [N x C x ...].
        :param t: 扩散步数.
        :param noise: 如果指定，则使用该高斯噪声.
        :return:x_start的一个带噪版本 x_t
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
        计算扩散后验 q(x_{t-1} | x_t, x_0) 的均值和方差
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )#均值
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)#方差
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        # assert (
        #     posterior_mean.shape[0]
        #     == posterior_variance.shape[0]
        #     == posterior_log_variance_clipped.shape[0]
        #     == x_start.shape[0]
        # )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x_t, y, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None
    ):
        """
        应用模型来获取 p(x_{t-1} | x_t) 的均值、方差，以及对 x_0 的预测

        :param x_t: 时间步t的带噪信号.
        :param y: 条件信号 (含噪雷达相位)
        :param t: 时间步.
        :param clip_denoised: 是否将去噪后的信号裁剪到[-1, 1]范围
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: 传递给模型的额外参数，例如条件 lq
        :return: 包含均值、方差和x_0预测值的字典
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        #采样（逆过程）单步逻辑顺序：
        #你有当前 x_t 和条件 y, timestep 索引 t.
        #归一化输入 + 前向网络 → model_output（预测某种语义）
        #用 model_output → 还原 pred_xstart
        #用解析公式 + pred_xstart → 后验均值 model_mean
        #固定的 model_variance / model_log_variance
        #采样得到x_(t-1)


    
        model_kwargs['lq'] = y 
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)#这两行获得的固定 model_variance / model_log_variance
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            return x
        # 根据模型的预测类型，计算对x_0的预测     # predict x_0
        pred_xstart = process_xstart(model_output)                                               #  predict \eps

        # 使用预测的x_0来计算x_{t-1}的均值
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }



    def p_sample(self, model, x, y, t, clip_denoised=False, model_kwargs=None):#这是 逆扩散单一步 由当前的x-t采样得到x_(t-1)
        """
        从模型在给定时间步t采样x_{t-1}.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            y,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop( 
        self,
        y,
        model,
        noise=None,
        clip_denoised=False,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        从模型生成样本（完整的去噪过程）

        :param y: 条件信号 (含噪雷达相位).
        :param model:  RDT模型.
        :param first_stage_model: the autoencoder model
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return:一个批次的去噪后样本.
        """
        final = None
        for sample in self.p_sample_loop_progressive( #调用 p_sample_loop_progressive 逐步生成；不断覆盖 final，循环结束后 final 就是最早时间（t=0）的 结果。
            y,
            model,
            noise=noise,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample["sample"]
        return final

    def p_sample_loop_progressive( #生成器：逆序遍历所有时间步，逐步调用 p_sample 并 yield 中间结果（可视化用）
            self, y, model,
            noise=None,
            clip_denoised=False,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
         逐步生成样本,并可以yield出每个中间步骤的结果

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device

        # generating noise
        if noise is None: #初始噪声：如未传 noise，随机生成（shape 与 z_y 相同）。noise_repeat 用于把同一噪声复制给整个 batch——多用于对比
            noise = th.randn_like(y)

        x_sample = self.prior_sample(y, noise) #构造初始状态

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm #时间步序列：倒序 T-1 → 0。若 progress=True，用 tqdm 包装显示进度条

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * y.shape[0], device=device) #组装 batch 时间索引张量 t
            with th.no_grad():
                out = self.p_sample( #调 p_sample(...) 得到：{"sample": x_{t-1}, "pred_xstart": ..., "mean": ...}
                    model,
                    x_sample,
                    y,
                    t,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs
                )
                yield out #yield 当前结果（允许外部在每一步做可视化 / 保存）
                x_sample = out["sample"] #更新 z_sample 为新状态，继续下一步

    def prior_sample(self, y, noise=None): #构造逆扩散初始点x_(T-1)（最高噪声层），中心在条件y的 latent 上，而不是纯高斯 与传统 DDPM 的初始化（纯高斯 x_T ~ N(0,I)) 不同——这里以条件 y 为“中心”（条件化起点），更像一次“从 y 出发的噪声包裹”。
        """
        Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)
        从先验分布q(x_T|y)中采样，作为去噪过程的起点
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """
        if noise is None:
            noise = th.randn_like(y)

        t = th.tensor([self.num_timesteps-1,] * y.shape[0], device=y.device).long()

        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def training_losses(
            self, model, x_start, y, t,
            model_kwargs=None,
            noise=None,
            ):
        """
        计算单个时间步的训练损失
        :param first_stage_model: autoencoder model
        :param x_start:纯净信号真值 (纯净呼吸)
        :param y: 条件信号 (含噪雷达相位)
        :param t: 时间步
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: 包含损失项的字典
        """
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None: #为本次随机时间步构造高斯噪声 ε（与 x_start 形状一致）  可外部传固定 noise 做 “重现实验” 或 consistency regularization。
            noise = th.randn_like(x_start)

        x_t = self.q_sample(x_start, y, t, noise=noise) #对应前向公式 其中 x_0 == x_start t 是批次中每个样本单独的时间步索引，形状 [B]；q_sample 内部用 _extract_into_tensor 广播η_t

        terms = {} #用来存放各种损失项（当前只有 "mse"；也可以后续加 "perceptual"、"freq" 等）

        model_kwargs['lq'] = y
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs) 

        target =  x_start
        assert model_output.shape == target.shape == x_start.shape #保证输出 & 监督 & 原始 latent 在形状上完全对齐（防止 silent broadcast 错误）
        terms["mse"] = mean_flat((target - model_output) ** 2) 
      

        if self.loss_type == LossType.WEIGHTED_MSE:
                weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
                terms["mse"] *= weights


       
        pred_zstart = model_output


        return terms, x_t, pred_zstart #terms: 含有 "mse"（shape [B]）。训练脚本一般会做 .mean() 作为最终 loss    x_t: 方便外部如果还想做某些与中间状态相关的约束（如 consistency regularization）   pred_zstart: 可用于重建指标 (PSNR, SSIM, 自定义频谱)、可视化

    def _scale_input(self, inputs, t): 

        return inputs

