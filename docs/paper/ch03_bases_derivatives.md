## 장 3. 기저 함수와 해석 도함수

### 3.1 좌표/곡률 변환 요약

정규화 좌표 $(u,v)\to(x,y)\to(\rho,\psi)$, 곡률 변환:
$$\rho'=\frac{\tanh(\kappa\rho)}{\tanh(\kappa)},\quad x'=\frac{\rho'}{\rho}x,\ y'=\frac{\rho'}{\rho}y.$$

### 3.2 Polynomial(1차)
$$f_{poly}(x',y';\theta)=x'\cos\theta+y'\sin\theta,$$
$$\partial_\theta f=-x'\sin\theta+y'\cos\theta,\quad \nabla f=(\cos\theta,\sin\theta).$$

### 3.3 Sinusoid(2D)
$$k=2\pi f_{norm}k_{max}(\cos\theta,\sin\theta),\ \Phi=k_x u+k_y v+\phi,\ g=\sin\Phi,$$
$$\partial_\phi g=\cos\Phi,\quad \partial_\theta g=\cos\Phi\,( -K\sin\theta\,u+K\cos\theta\,v).$$

### 3.4 RBF-Gaussian
$$g=\exp\!\Big(-\tfrac{x'^2+y'^2}{2\sigma^2}\Big),\quad \nabla_{x',y'} g=-(g/\sigma^2)(x',y').$$

### 3.5 tanh-bump
$$g=\tanh\big(\beta(1-r'^2)\big),\quad \nabla g=-2\beta\,\mathrm{sech}^2\big(\beta(1-r'^2)\big)(x',y').$$

### 3.6 Bessel J0 (선택)
$$g=J_0(\xi),\ \xi=\omega r',\quad \partial_{r'} g=-\omega J_1(\xi).$$


