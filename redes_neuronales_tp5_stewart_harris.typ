
#import "@preview/basic-report:0.3.1": *
#import "@preview/dashy-todo:0.1.2": todo
#import "@preview/subpar:0.2.2"

#let todo-inline(content) = {
  todo(position: "inline")[#content]
}

#let w0p = $w_0^"patrón"$
#let w1p = $w_1^"patrón"$

#let w0est = $w_0^"est"$
#let w1est = $w_1^"est"$

#set math.equation(numbering: none)

#show: it => basic-report(
  doc-category: [Redes neuronales #emoji.brain],
  doc-title: "Trabajo práctico 5: \nRegresión y clasificación",
  author: "María Luz Stewart Harris",
  affiliation: "Instituto Balseiro",
  language: "es",
  compact-mode: true,
  heading-font: "Vollkorn",
  heading-color: black,
  it
)

= Ej 1.: Regresión lineal

El objetivo del trabajo es generar un modelo basado en regresión lineal para estimar el peso de una persona en función de su altura: 
$
  W = w_0 + w_1 dot H
$
donde 
- $W$: peso estimado de la persona.
- $H$: altura de la persona.
- $w_0, w_1$: parámetros del modelo.


Para entrenar el modelo, se utilizan datos de mediciones de peso y altura de 25000 personas.

== Cálculo de parámetros

A partir de un conjunto de datos de peso y altura se calculan los parámetros $w_0, w_1$ y los siguientes parámetros:
- $R S S$: suma residual de cuadrados del valor de peso estimado.
- $sigma^2$: varianza estimada del error entre el valor de peso estimado y el valor real.
- $S E^2$: varianza para cada parámetro.

```python
def linear_regression(x: np.ndarray, y: np.ndarray):
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    w1 = np.sum(x_centered*y_centered) / np.sum(x_centered**2)
    w0 = y_mean - w1 * x_mean
    return w0, w1
```

```python
def calc_params(h:np.ndarray, w:np.ndarray):
    N = len(h)
    w0, w1 = linear_regression(h, w)
    w_est = w0 + w1 * h
    RSS = np.sum((w-w_est)**2)
    TSS = np.sum((w-w.mean())**2)
    σ_sqr = RSS/(N-2)
    SE_sqr_w0 = σ_sqr * (1/N + h.mean()**2/np.sum((h-h.mean())**2))
    SE_sqr_w1 = σ_sqr * (1/np.sum((h-h.mean())**2))
    return w0, w1, RSS, TSS, σ_sqr, SE_sqr_w0, SE_sqr_w1
```

== Error en el cálculo de parámetros

Se calculan $w_0, w_1$ utilizando una regresión lineal de un subconjunto de los 25000 datos de entrenamiento disponibles, y luego se estudia el error en la estimación de los parámetros en función de la cantidad de datos elegidos.

En primer lugar, se calcula $w_0, w_1$ con un conjunto de $N=25000$ datos (es decir, todos los datos de entrenamiento disponibles). Estos valores son consideras  "patrón", es decir, se toman como el valor real#footnote([Se consideran "reales" en el sentido de que son los valores estimados con menor error posible basados en los datos de entrenamiento disponibles, pero no dejan de ser una estimación.]) de los parámetros contra el cual se deben hacer comparaciones de otras estimaciones de $w_0, w_1$ con $N < 25000$:

$
  w0p &= -&&82.576 \
  w1p &= &&3.083 \
$

Luego, para N entre 10 y 24500, se repite 1000 veces:
  - Se toman N muestras de los datos de entrenamiento.
  - Se calculan #w0est, #w1est, $S E_1$, $S E_0$.
  - Se calculan los intervalos de confianza 
    - $I C_0 = [w0est -2 S E_0" "; " "w0est + 2 S E_0]$
    - $I C_1 = [w1est -2 S E_1" "; " "w1est + 2 S E_1]$

La @w_164 muestra los resultados obtenidos para $N=164$. Los valores obtenidos para #w0est y #w1est siguen aproximadamente una disctribución normal (@w_hist_164). La @w_IC_164 muestra los 1000 intervalos de confianza obtenidos. En negro se grafican los intervalos de confianza que contienen el valor real, y en rojo los que no lo contienen.

#subpar.grid(
  columns: (1fr),
  figure(image("figures/w_n_162_reps_1000.png", width: 70%), caption: [Histogramas de #w0est, #w1est, para N=164 y 1000 repeticiones. ]), <w_hist_164>,
  figure(image("figures/w_CI_n_162_reps_1000.png"), caption: [ $I C_0, I C_1$ para N=164 y 1000 repeticiones. Las líneas horizontales marcan #w0p, #w1p.]), <w_IC_164>,

  caption: [],
  label: <w_164>
)

Al aumentar $N$, $I C_0$ y $I C_1$ reducen su amplitud y se centran en #w0p, y #w1p respectivamente. En otras palabras, el error de la estimación se reduce al aumentar $N$ (@w_CI_single_rep).

#subpar.grid(
  columns: (1fr),
  figure(image("figures/w_CI_single_rep_0.png"), caption: [Primera repetición.]), <w_CI_single_rep_0>,
  figure(image("figures/w_CI_single_rep_1.png"), caption: [Segunda repetición.]), <w_CI_single_rep_1>,

  caption: [$I C_0$ y $I C_1$ en función de $N$ para dos repeticiones distintas. ],
  label: <w_CI_single_rep>
)

La relación entre las amplitudes promedio de los intervalos de confianza y la cantidad de datos tomados sigue la ley de potencias (ver @SE, y observar la relación lineal con escalas logaritmicas en ambos ejes).

Las pendientes de los gráficos son:
$
  (Delta log(S E_0,1))/(Delta log(N)) =  alpha approx -0.5
$

A partir de las pendientes de los gráficos, se obtiene:

$
  S E_(0,1) prop 1/sqrt(N)
$

#subpar.grid(
  columns: (1fr),
  figure(image("figures/SE_single_rep.png"), caption: [Primera iteración.]), <SE_single_rep_0>,
  figure(image("figures/SE_mean_reps_1000.png"), caption: [Promedio (1000 repeticiones).]), <SE_mean>,

  caption: [$S E$ en función de $N$ para una iteración y promedio. ],
  label: <SE>
)

Por lo tanto, se puede controlar la precisión promedio de los parámetros a través de la elección de cantidad de muestras. La ecuación anterior permite elegir la cantidad de muestras necesarias teniendo en cuenta la precisión de predicción necesaria para la aplicación específica del modelo.


= Anexo

El repostorio de código utilizado para simular y graficar se encuentra en:
https://github.com/malustewart/redes-neuronales-tp5