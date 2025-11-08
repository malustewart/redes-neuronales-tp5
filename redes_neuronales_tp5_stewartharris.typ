
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

#todo-inline([histogramas w0, w1 para n=10,1000, etc, N=1000])

#todo-inline([IC para todas las reps de un n, para varios n. Comentar que en el ~95% de las repeticiones el SE contiene el w golden.])

#todo-inline([IC para todas una rep (o el promedio de todas?) de todos los n. Comentar que se reduce SE])


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
