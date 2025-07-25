# 제1장: RBE 수학적 기초 및 푸앵카레볼 기하학

## 1.1 서론

Riemannian Basis Encoding (RBE)는 신경망 가중치의 극한 압축을 위한 수학적 프레임워크입니다. 본 장에서는 RBE의 핵심인 푸앵카레볼 기하학의 수학적 기초를 엄밀하게 정립합니다.

## 1.2 쌍곡기하학의 기본 개념

### 1.2.1 쌍곡공간의 정의

$n$차원 쌍곡공간 $\mathbb{H}^n$은 상수 음의 곡률 $-1$을 갖는 완전한 단순연결 리만 다양체입니다.

**정의 1.1** (쌍곡공간)
$$\mathbb{H}^n = \{x \in \mathbb{R}^{n+1} : \langle x, x \rangle_L = -1, x_0 > 0\}$$

여기서 $\langle \cdot, \cdot \rangle_L$는 로렌츠 내적:
$$\langle x, y \rangle_L = -x_0 y_0 + x_1 y_1 + \cdots + x_n y_n$$

### 1.2.2 푸앵카레볼 모델

**정의 1.2** (푸앵카레볼)
$$\mathcal{D}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$$

푸앵카레볼은 쌍곡공간 $\mathbb{H}^n$의 등각 모델(conformal model)입니다.

**메트릭 텐서**:
$$g_{ij} = \frac{4\delta_{ij}}{(1-\|x\|^2)^2}$$

따라서 선소 $ds$는:
$$ds^2 = \frac{4}{(1-\|x\|^2)^2} \sum_{i=1}^n dx_i^2$$

## 1.3 2차원 푸앵카레볼의 극좌표 표현

RBE에서는 2차원 푸앵카레볼을 사용하므로, 극좌표 $(r, \theta)$로 표현합니다.

### 1.3.1 좌표 변환

**직교좌표에서 극좌표로**:
$$x = r\cos\theta, \quad y = r\sin\theta$$

여기서 $0 \leq r < 1$, $0 \leq \theta < 2\pi$

### 1.3.2 극좌표에서의 메트릭

직교좌표에서의 메트릭:
$$ds^2 = \frac{4}{(1-r^2)^2}(dx^2 + dy^2)$$

극좌표 변환을 적용하면:
$$\frac{\partial x}{\partial r} = \cos\theta, \quad \frac{\partial x}{\partial \theta} = -r\sin\theta$$
$$\frac{\partial y}{\partial r} = \sin\theta, \quad \frac{\partial y}{\partial \theta} = r\cos\theta$$

따라서:
$$dx^2 + dy^2 = dr^2 + r^2 d\theta^2$$

**극좌표에서의 푸앵카레 메트릭**:
$$ds^2 = \frac{4}{(1-r^2)^2}(dr^2 + r^2 d\theta^2)$$

### 1.3.3 메트릭 텐서의 성분

$$g_{rr} = \frac{4}{(1-r^2)^2}, \quad g_{\theta\theta} = \frac{4r^2}{(1-r^2)^2}, \quad g_{r\theta} = 0$$

**메트릭 텐서의 역행렬**:
$$g^{rr} = \frac{(1-r^2)^2}{4}, \quad g^{\theta\theta} = \frac{(1-r^2)^2}{4r^2}, \quad g^{r\theta} = 0$$

## 1.4 푸앵카레볼에서 유클리드 공간으로의 매핑

### 1.4.1 쌍곡 거리 함수

푸앵카레볼의 점 $(r, \theta)$를 쌍곡 거리로 변환하는 함수를 정의합니다.

**정리 1.1** (쌍곡 거리 변환)
푸앵카레볼의 점 $r \in [0, 1)$에 대해 쌍곡 거리 $d$는:
$$d = 2 \tanh^{-1}(r)$$

**증명**:
푸앵카레볼에서 원점으로부터의 거리는:
$$d = \int_0^r \frac{2}{1-t^2} dt$$

$u = \tanh(s/2)$로 치환하면 $du = \frac{1}{2}\text{sech}^2(s/2)ds = \frac{1-u^2}{2}ds$

따라서 $ds = \frac{2}{1-u^2}du$

$t = 0$일 때 $s = 0$, $t = r$일 때 $s = 2\tanh^{-1}(r)$

$$d = \int_0^{2\tanh^{-1}(r)} ds = 2\tanh^{-1}(r)$$

### 1.4.2 경계 조건 처리

$r \to 1$일 때 $\tanh^{-1}(r) \to \infty$이므로 수치적 안정성을 위해:

$$d = \begin{cases}
2\tanh^{-1}(r) & \text{if } r < 0.999 \\
2 \cdot \frac{1}{2}\ln\left(\frac{1+r}{1-r}\right) & \text{if } r \geq 0.999
\end{cases}$$

여기서 $\tanh^{-1}(r) = \frac{1}{2}\ln\left(\frac{1+r}{1-r}\right)$의 항등식을 사용했습니다.

## 1.5 RBE 기본 함수

### 1.5.1 핵심 변환 함수

RBE의 핵심 함수는 다음과 같이 정의됩니다:

**정의 1.3** (RBE 기본 함수)
$$f(r, \theta, i, j) = \tanh(d) \cdot \sin(\theta) \cdot m(i, j)$$

여기서:
- $d = 2\tanh^{-1}(r)$: 쌍곡 거리
- $\sin(\theta)$: 각도 성분
- $m(i, j)$: 위치 기반 변조 함수

### 1.5.2 위치 기반 변조 함수

**정의 1.4** (공간 변조)
$$m(i, j) = 1 + 0.1 \cdot \sin\left(2\pi \cdot \frac{h(i, j)}{256}\right)$$

여기서 $h(i, j) = (31i + 17j) \bmod 256$는 위치 해시 함수입니다.

**수학적 정당성**:
1. **균등 분포**: 선형 합동 생성기의 성질에 의해 $(i, j)$ 위치에 대해 균등 분포
2. **결정론적**: 동일한 $(i, j)$에 대해 항상 같은 값
3. **국소 변동**: 인접한 위치에서 다른 값으로 행렬의 다양성 확보

### 1.5.3 합성 함수의 성질

**정리 1.2** (RBE 함수의 연속성)
$f(r, \theta, i, j)$는 $(r, \theta) \in [0, 1) \times [0, 2\pi)$에서 연속이고 미분가능합니다.

**증명**:
1. $\tanh(2\tanh^{-1}(r))$는 $r \in [0, 1)$에서 연속이고 미분가능
2. $\sin(\theta)$는 $\theta \in [0, 2\pi)$에서 연속이고 미분가능
3. $m(i, j)$는 고정된 $(i, j)$에 대해 상수이므로 연속
4. 연속함수들의 곱은 연속

## 1.6 함수의 미분 계산

### 1.6.1 r에 대한 편미분

$$\frac{\partial f}{\partial r} = \frac{\partial}{\partial r}[\tanh(2\tanh^{-1}(r))] \cdot \sin(\theta) \cdot m(i, j)$$

**단계 1**: $\frac{d}{dr}[2\tanh^{-1}(r)]$ 계산

$$\frac{d}{dr}[2\tanh^{-1}(r)] = 2 \cdot \frac{1}{1-r^2}$$

**단계 2**: $\frac{d}{dr}[\tanh(2\tanh^{-1}(r))]$ 계산

$u = 2\tanh^{-1}(r)$라 하면:
$$\frac{d}{dr}[\tanh(u)] = \text{sech}^2(u) \cdot \frac{du}{dr} = (1-\tanh^2(u)) \cdot \frac{2}{1-r^2}$$

$\tanh(2\tanh^{-1}(r)) = r$이므로:
$$\frac{d}{dr}[\tanh(2\tanh^{-1}(r))] = (1-r^2) \cdot \frac{2}{1-r^2} = 2$$

**최종 결과**:
$$\frac{\partial f}{\partial r} = 2 \cdot \sin(\theta) \cdot m(i, j)$$

### 1.6.2 θ에 대한 편미분

$$\frac{\partial f}{\partial \theta} = \tanh(2\tanh^{-1}(r)) \cdot \cos(\theta) \cdot m(i, j)$$

$\tanh(2\tanh^{-1}(r)) = r$이므로:
$$\frac{\partial f}{\partial \theta} = r \cdot \cos(\theta) \cdot m(i, j)$$

## 1.7 리만 기하학적 그래디언트

### 1.7.1 자연 그래디언트의 정의

일반적인 유클리드 그래디언트 대신 리만 메트릭을 고려한 자연 그래디언트를 사용합니다.

**정의 1.5** (자연 그래디언트)
$$\nabla_g f = g^{-1} \nabla f$$

여기서 $g^{-1}$는 메트릭 텐서의 역행렬입니다.

### 1.7.2 푸앵카레볼에서의 자연 그래디언트

메트릭 역행렬:
$$g^{-1} = \frac{(1-r^2)^2}{4} \begin{pmatrix} 1 & 0 \\ 0 & \frac{1}{r^2} \end{pmatrix}$$

유클리드 그래디언트:
$$\nabla f = \begin{pmatrix} \frac{\partial f}{\partial r} \\ \frac{\partial f}{\partial \theta} \end{pmatrix}$$

**자연 그래디언트**:
$$\nabla_g f = \frac{(1-r^2)^2}{4} \begin{pmatrix} \frac{\partial f}{\partial r} \\ \frac{1}{r^2} \frac{\partial f}{\partial \theta} \end{pmatrix}$$

### 1.7.3 손실 함수와의 연쇄 법칙

손실 함수 $L = \frac{1}{2}(f - t)^2$에 대해:
$$\frac{\partial L}{\partial f} = f - t$$

따라서 최종 그래디언트는:
$$\frac{\partial L}{\partial r} = (f - t) \cdot \frac{(1-r^2)^2}{4} \cdot 2 \sin(\theta) m(i, j)$$
$$\frac{\partial L}{\partial \theta} = (f - t) \cdot \frac{(1-r^2)^2}{4r^2} \cdot r \cos(\theta) m(i, j)$$

## 1.8 수치적 안정성

### 1.8.1 경계 근처에서의 안정성

$r \to 1$일 때 다음 문제들이 발생합니다:
1. $(1-r^2)^{-1} \to \infty$
2. $\tanh^{-1}(r) \to \infty$

**해결책 1**: 경계 클리핑
$$r_{\text{safe}} = \min(r, 0.999)$$

**해결책 2**: 적응적 그래디언트 클리핑
$$\text{clip\_factor} = \max(1-r^4, 0.01)$$

### 1.8.2 0 나누기 방지

$r = 0$ 근처에서 $\frac{1}{r^2}$ 항이 발산하므로:
$$\text{safe\_r} = \max(r, 10^{-6})$$

## 1.9 고정소수점 표현

### 1.9.1 Q64 고정소수점 형식

64비트 정수로 $[0, 1)$ 실수를 표현:
$$r_{\text{fixed}} = \lfloor r \cdot 2^{64} \rfloor$$

**정밀도**: $2^{-64} \approx 5.4 \times 10^{-20}$

### 1.9.2 각도의 고정소수점 표현

$\theta \in [0, 2\pi)$를 64비트로 표현:
$$\theta_{\text{fixed}} = \lfloor \frac{\theta}{2\pi} \cdot 2^{64} \rfloor$$

## 1.10 정리

본 장에서는 RBE의 수학적 기초인 푸앵카레볼 기하학을 엄밀하게 정립했습니다. 핵심 결과들:

1. **메트릭 공식**: $ds^2 = \frac{4}{(1-r^2)^2}(dr^2 + r^2 d\theta^2)$
2. **기본 함수**: $f(r, \theta, i, j) = \tanh(2\tanh^{-1}(r)) \sin(\theta) m(i, j)$
3. **자연 그래디언트**: 리만 메트릭을 고려한 안정적인 최적화
4. **수치 안정성**: 경계 조건과 0 나누기 방지 기법

다음 장에서는 이 기하학적 구조를 실제 신경망 가중치 압축에 적용하는 방법을 다룹니다. 