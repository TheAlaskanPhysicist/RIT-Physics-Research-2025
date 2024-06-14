
Thomas Baumgarte (1) - Numerical relativity: Mathematical formulation
https://www.youtube.com/watch?v=t3uo2R-yu4o&list=PLRVOWML3TL_djTd_nsTlq5aJjJET42Qke

Notes:


### Newtonian Gravity
Central object potential $\phi$.
Freely-falling object experience (acceleration):
$$\vec a=-\vec\nabla\phi \iff a_i=-\nabla_i\phi$$
Distance between nearby freely-falling objects is related to the 2nd derivatives of $\phi$:
$$\sim\nabla_i\nabla_j\phi$$
This is identical to the geodesic deviation (since they attract)

Potential $\phi$ satisfies the Laplace equations, the trace of the second derivative of $\phi$.

Newtonian field equation:
$$\nabla^2\phi = \nabla^i\nabla_i\phi=4\pi\rho$$

Notice: 
 - A central object
 - The geodesics, motion of freely-falling objects (related to the first derivative)
 - The geodesic deviation, distance between freely-falling objects (related to the second derivative)
 - Taking the trace of the second derivatives, object involved in the field equation.

Note: "gauge freedom", where: $\phi \rightarrow \phi + c$


### General Relativity
Central Object: Spacetime metric $g_{ab}$
Use $g_{ab}$ to measure proper distance $ds^2=g_{ab}dx^adx^b$  (distances between spacetime events -> proper distances)

How to measure curvature?
- Sum of angles in triangles.
- Parallel transport of vectors.

#### Covariant Derivative
Generalizes the partial derivative so that the result is a tensor.
Its components are:
$$\nabla_aT^b = \partial_aT^b+T^c\Gamma^{b}_{ac}$$

In a coordinate basis, the connection coefficients are Christoffel Symbols.
$$\Gamma^a_{bc}=\dfrac{1}{2}g^{a\alpha}\left(\partial_c+...\right)$$


Freely-falling objects parallel transport their own 4-velocity.
-> They follow Geodesics.

$\nabla_aU^b=\partial_aU^b+U^c\Gamma^{b}_{ac}=0$
(Still involves first-derivatives)

Geodesic deviation: Expects second derivatives of the metric
-> Leads to the Riemann Tensor (effectively second derivatives of the metric)

#### Riemann Tensor
Measures curvature, we define it as the commutator of path transport ordering.





LAST at 23:36