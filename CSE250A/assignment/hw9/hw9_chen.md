# HW9 CSE250 SectionA

Name: Guanghao Chen

PID: A53276390

Email: guc001at eng.ucsd.edu

## 9.1 Efficient horizon time

$$
\begin{align}\sum_{n\ge t}\gamma^nr_n&\le\sum_{n\ge t}\gamma^n\\
&=\frac{\gamma^t(1-\gamma^{\infty})}{1-\gamma}\\
&=\frac{\gamma^t}{1-\gamma}\\
log\gamma&\le\gamma-1\\
\gamma&\le e^{\gamma-1}\\
\gamma^t&\le e^{t(\gamma-1)}\\
\therefore\frac{\gamma^t}{1-\gamma}&\le\frac{e^{t(\gamma-1)}}{1-\gamma}\\
&=he^{-t/h}
\end{align}
$$

## 9.2 Three State, two action MDP

### (a) Policy evaluation

Notice the three actions are up, down and down.

Therefore, the state value should be
$$
(\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}-\frac{3}{4}\begin{bmatrix}\frac{1}{3}&\frac{2}{3}&0\\\frac{2}{3}&\frac{1}{3}&0\\0&\frac{2}{3}&\frac{1}{3}\end{bmatrix})^{-1}\times\begin{bmatrix}24\\-6\\12\end{bmatrix}=\begin{bmatrix}48\\24\\32\end{bmatrix}
$$

### (b) Greedy policy

$\pi^{'}(1)$
$$
\begin{bmatrix}\frac{1}{3}&0&\frac{2}{3}\end{bmatrix}\times\begin{bmatrix}48\\24\\32\end{bmatrix}=37.3\\
\begin{bmatrix}\frac{1}{3}&\frac{2}{3}&0\end{bmatrix}\times\begin{bmatrix}48\\24\\32\end{bmatrix}=32
$$
$\pi^{'}(2)$
$$
\begin{bmatrix}\frac{2}{3}&\frac{1}{3}&0\end{bmatrix}\times\begin{bmatrix}48\\24\\32\end{bmatrix}=40\\
\begin{bmatrix}0&\frac{1}{3}&\frac{2}{3}\end{bmatrix}\times\begin{bmatrix}48\\24\\32\end{bmatrix}=29.33
$$
$\pi^{'}(3)$
$$
\begin{bmatrix}0&\frac{2}{3}&\frac{1}{3}\end{bmatrix}\times\begin{bmatrix}48\\24\\32\end{bmatrix}=26.6667\\
\begin{bmatrix}\frac{2}{3}&0&\frac{1}{3}\end{bmatrix}\times\begin{bmatrix}48\\24\\32\end{bmatrix}=42.6667
$$
**Therefore, the second column of the table is $\downarrow,\downarrow,\uparrow$.**

## 9.3 Value function fora random walk

### (a)

Although there're infinite states from 0 to $\infty$, only when $s^{'}=s$ and $s^{'}=s+1$ the transition probabilities are non-zero.

Therefore, 
$$
\begin{align}V^\pi(s) &= R(s) + \gamma \sum_{s'=0}^{\infty}P(s'|s, \pi(s))V^\pi(s')\\
	&= R(s) + \gamma \sum_{s'=s}^{s+1}P(s'|s, \pi(s))V^\pi(s')\end{align}
$$

### (b)

Substituting $V^{\pi}(s) =as +b$ into the Bellman equation, it has
$$
\begin{align}as+b&=s+\gamma[\frac{2}{5}(as+b)+\frac{3}{5}(a(s+1)+b)]\\
as+b&=s+\gamma[\frac{2}{5}(as+b)+\frac{3}{5}(as+a+b)]\\
as+b&=s+\gamma[as+\frac{3}{5}a+b]\\
((1-\gamma)a-1))s&=\frac{3}{5}\gamma a-(1-\gamma)b\\
\end{align}
$$
Requiring both sides are equal for all values of s,
$$
\left\{\begin{align}a&=\frac{1}{1-\gamma}\\
b&=\frac{3\gamma}{5(1-\gamma)^2}
\end{align}\right.
$$

##9.4 Policy and value iteration

See the attachment.

## 9.5 Convergence of iterative policy evaluation

$$
\begin{align}
\Delta k &= max_s|V_k(s)-V^{\pi}(s)|\\
&=max_s|R(s)+\gamma\sum_{s^{'}}P(s^{'}|s,\pi(s))V_{k-1}(s)-R(s)-\gamma\sum_{s^{'}}P(s^{'}|s,\pi(s))V^{\pi}(s)|\\
&=max_s|\gamma\sum_{s^{'}}P(s^{'}|s,\pi(s))V_{k-1}(s)-\gamma\sum_{s^{'}}P(s^{'}|s,\pi(s))V^{\pi}(s)|\\
&=max_s\gamma|\sum_{s^{'}} P(s^{'}|s,\pi(s))(V_{k-1}(s)-V^{\pi}(s))|\\
&= max_s\gamma\sum_{s^{'}} P(s^{'}|s,\pi(s))|V_{k-1}(s)-V^{\pi}(s)|\\
&\le max_s\gamma\sum_{s^{'}}P(s^{'}|s,\pi(s))\Delta_{k-1}\\
&=\gamma\Delta_{k-1}\\
&=\gamma^{k}\Delta_{1}\\
\end{align}
$$

## 9.6 Stochastic approximation

### (a)

$$
\begin{align}
\sum_{k=1}^{\infty} \alpha_k &= 1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \frac{1}{5} + ... + \frac{1}{8} + ...\\
	&\geq 1 + \frac{1}{2} + (\frac{1}{4} + \frac{1}{4}) + (\frac{1}{8} + \frac{1}{8} + \frac{1}{8} + \frac{1}{8}) + ...\\
	&= 1 + \frac{1}{2} + \frac{1}{2} + \frac{1}{2} + ...\\
	&= \infty\\
\sum_{k=1}^{\infty} \alpha_k^2 &= 1 + \frac{1}{2 \times 2} + \frac{1}{3 \times 3}  + \frac{1}{4 \times 4} + ...\\
	&\leq 1 + \frac{1}{1 \times 2} + + \frac{1}{2 \times 3} + + \frac{1}{3 \times 4} + ...\\
	&= 1 + (1 - \frac{1}{2}) + (\frac{1}{2} - \frac{1}{3}) + (\frac{1}{3} - \frac{1}{4}) + ...\\
	&\leq 2 
\end{align}
$$

### (b)

Suppose $\mu_k=\frac{1}{k}(x_1+x_2+...+x_k)$.

According to the update rule,
$$
\begin{align}\mu_{k+1} &= \mu_k + \frac{1}{k+1}(x_{k+1} - \mu_k)\\
&= \frac{1}{k} (x_1 + x_2 + ... + x_k) + \frac{1}{k+1} [x_{k+1} - \frac{1}{k} (x_1 + x_2 + ... + x_k)]\\
&= \frac{k+1-1}{k(k+1)}(x_1 + x_2 + ... + x_k) + \frac{1}{k+1} x_{k+1}\\
&= \frac{1}{k+1}(x_1 + x_2 + ... + x_k) + \frac{1}{k+1} x_{k+1}\\
&= \frac{1}{k+1}(x_1 + x_2 + ... + x_k + x_{k+1})
\end{align}
$$



