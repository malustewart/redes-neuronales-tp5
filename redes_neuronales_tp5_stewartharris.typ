
#import "@preview/basic-report:0.3.1": *
#import "@preview/dashy-todo:0.1.2": todo

#set math.equation(numbering: "(1)")

#let todo-inline(content) = todo(position:"inline")[#content]

#show: it => basic-report(
  doc-category: "Redes neuronales",
  doc-title: "Trabajo práctico 5: \nRegresión y clasificación",
  author: "María Luz Stewart Harris",
  affiliation: "Instituto Balseiro",
  language: "es",
  compact-mode: true,
  it
)

= Ej. 1: Regresión lineal

#todo-inline([Como elijo los golden])

#todo-inline([Codigo de la regresion lineal])

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
def calc_predictors(h:np.ndarray, w:np.ndarray):
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

#todo-inline([histogramas w0, w1 para n=10,1000, etc, N=1000])

#todo-inline([IC para todas las reps de un n, para varios n. Comentar que en el ~95% de las repeticiones el SE contiene el w golden.])

#figure(
    grid(
      columns: 1,
      rows: 2,
      gutter: 0mm,
      image("figures/w_CI_n_25_reps_1000.png"),
      image("figures/w_CI_n_1000_reps_1000.png"),
    ),
  caption: "Intervalos de confianza (0.95) para las 1000 repeticiones a diferentes tamaños de muestras."
)<fig:w_CI_multiple_reps>

#todo-inline([Agregar en la descripcion de los graficos que la linea horizontal es el valor "real"])

#todo-inline([IC para todas una rep (o el promedio de todas?) de todos los n. Comentar que se reduce SE. En los graficos poner el area sombreada con fill_between])

#todo-inline([Conclusion: hace falta tantos datos?])

// ```python
// def create_random_memories(N,p):
//     return np.random.choice(a=[-1, 1], size=(p,N))
// ```

// #figure(
//   image("figs/conv_times_N_3000_alfa_0.1.png"),
//   caption: "Tiempo de convergencia para modelo de Hopfield sin ruido con N=3000, α=0.1 (máximo 10 iteraciones de todo el sistema). Los casos marcados como \"No converge\" son aquellos en los que el sistema oscilaba entre dos estados."
// )<fig:t_convergencia>

// #figure(
//     grid(
//       columns: 2,
//       rows: 2,
//       gutter: 0mm,
//       image("figs/overlap_histogram_N_500.png"),
//       image("figs/overlap_histogram_N_1000.png"),
//       image("figs/overlap_histogram_N_2000.png"),
//       image("figs/overlap_histogram_N_4000.png"),
//     ),
//   caption: "Histogramas del overlap entre el punto inicial y el punto de convergencia, tomando como puntos iniciales todos los patrones almacenados."
// )<fig:overlaps_fixed_N>

= Anexo
El repostorio de código utilizado para simular y graficar se encuentra en: #box[#link( "https://github.com/malustewart/redes-neuronales-tp5" )]
