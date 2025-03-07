FIRRTL version 3.3.0
circuit NestedBundleModule :
  module NestedBundleModule : @[src/main/scala/gcd/NestedWhen.scala 61:7]
    input clock : Clock @[src/main/scala/gcd/NestedWhen.scala 61:7]
    input reset : UInt<1> @[src/main/scala/gcd/NestedWhen.scala 61:7]
    output io : { g : { e : { c : { a : UInt<2>, flip b : UInt<3>[2]}[2], flip d : UInt<3>, e : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<2>}}, f : UInt<2>[3]}[2], flip h : { e : { c : { a : UInt<2>, flip b : UInt<3>[2]}[2], flip d : UInt<3>, e : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<2>}}, f : UInt<2>[3]}[2]} @[src/main/scala/gcd/NestedWhen.scala 62:14]

    reg reg : { e : { c : { a : UInt<2>, flip b : UInt<3>[2]}[2], flip d : UInt<3>, e : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<2>}}, f : UInt<2>[3]}[2], clock @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[0].f, io.h[0].f @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[0].e.e.bits, io.h[0].e.e.bits @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[0].e.e.valid, io.h[0].e.e.valid @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[0].e.e.ready, io.h[0].e.e.ready @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[0].e.d, io.h[0].e.d @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[0].e.c, io.h[0].e.c @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[1].f, io.h[1].f @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[1].e.e.bits, io.h[1].e.e.bits @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[1].e.e.valid, io.h[1].e.e.valid @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[1].e.e.ready, io.h[1].e.e.ready @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[1].e.d, io.h[1].e.d @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect reg[1].e.c, io.h[1].e.c @[src/main/scala/gcd/NestedWhen.scala 64:20]
    connect io.g, reg @[src/main/scala/gcd/NestedWhen.scala 65:8]

