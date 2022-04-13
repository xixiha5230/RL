# 策略梯度（Policy Gradient）

## 1、算法解析

策略$\pi$是一个参数为的$\theta$网络，输入为observation，输出为 action 的**概率**，如下图所示。

![img](https://s2.loli.net/2022/04/12/wKnAMYshDEWzLg2.jpg)

**轨迹（Trajectory）**是一个回合（episode）中每个状态以及对应状态下采取的动作所构成的序列，如下图所示。

![img](https://s2.loli.net/2022/04/12/BHKecXJout5p7bN.png)

轨迹$\tau$发生的概率为：
$$
\begin{eqnarray}
p_\theta(\tau)&=&p_\theta(s_1,a_1,...,s_T,a_T)\\
&=&p(s_1)p_\theta(a_1|s_1)p(s_2|s_1,a_1)*p_\theta(a_2|s_2)p(s_3|s_2,a_2)...*p_\theta(a_T|s_T)p(END|s_T,a_T)\\
&=&p(s_1)\prod_{t=1}^Tp_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)
\end{eqnarray}
$$
其中$p_\theta(a_t|s_t)$是策略部分，与参数$\theta$相关，$p(s_{t+1}|s_t,a_t)$是环境动力动力学部分，与$\theta$无关。

轨迹中加入**Reward**为：

![img](https://s2.loli.net/2022/04/12/umG9hMYBW17nXOo.jpg)

单条轨迹的**Reward**为：
$$
R(\tau)=\sum_tr_t=\sum_tr(s_t,a_t)
$$
策略的**Expected Reward**为:
$$
\bar{R}_\theta=\sum_{\tau}R(\tau)p_{\theta}(\tau)=E_{{\tau\sim}p_\theta(\tau)}[R(\tau)]
$$
因此，PG方法的目标是求得最优参数$\theta^*$使得 **Expected Reward** 最大化:
$$
\begin{eqnarray}
\theta^*&=&\arg\max_{\theta}\bar{R}_\theta\\
&=&\arg\max_{\theta}E_{{\tau\sim}p_\theta(\tau)}[\sum_tr(s_t,a_t)]\\
&=&\arg\max_{\theta}\sum_{t=1}^TE_{(s_t,a_t){\sim}p_{\theta}(s_t,a_t)}[r(s_t,a_t)]\\
\end{eqnarray}
$$
令$\bar{R}_\theta=J(\theta)$则目标函数为：
$$
\begin{eqnarray}
J(\theta)&=&\sum_{\tau}R(\tau)p_{\theta}(\tau)\\
&=&E_{{\tau\sim}p_\theta(\tau)}[R(\tau)]\\
&=&{\int}p_\theta(\tau)R(\tau)d\tau\\
通过多次采样，不需要知道p_\theta,也能近似得到:\\
&\approx&\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^Tr(s_t^n,a_t^n)
\end{eqnarray}
$$
对目标函数$J(\theta)$求导：
$$
{\nabla}_{\theta}J(\theta)=\int{{\nabla}_{\theta}p_\theta(\tau)R(\tau)}d\tau
$$
可得
$$
{\nabla}_{\theta}J(\theta)=E_{\tau{\sim}p_\theta(\tau)}[\nabla_\theta\sum_{t=1}^Tlogp_\theta(a_t|s_t)R(\tau)]
$$
通过多次采样
$$
{\nabla}_{\theta}J(\theta){\approx}\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^T{\nabla_\theta}logp_\theta(a_t^n|s_t^n)R(\tau^n)
$$
**每个step的LOSSS**为：
$$
LOSS=-logp_\theta(a_t|s_t)R(\tau)
$$
在PG中使用$G(\tau)$代替$R(\tau)$，

$G(\tau)$是表示在当前step之后能拿到的**reward**

PG代码如下：

```python
# 计算G(t)
running_add = 0
for i in reversed(range(steps)):
    if reward_pool[i] == 0:
        running_add = 0
        else:
            running_add = running_add * gamma + reward_pool[i]
            reward_pool[i] = running_add
# update网络
for i in range(steps):
    state = state_pool[i]
    action = Variable(torch.FloatTensor([action_pool[i]]))
    reward = reward_pool[i]

    probs = policy_net(state)
    m = Bernoulli(probs)
    loss = -m.log_prob(action) * reward  #这个reward是G(t)
    loss.backward()
```

$$
LOSS=-logp_\theta(a_t|s_t)G(\tau)
$$

