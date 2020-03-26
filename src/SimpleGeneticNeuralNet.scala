
import scala.collection.mutable
import org.nlogo.{agent, api, core, nvm}
import core.Syntax
import api.ScalaConversions._
import api.{Argument, Context, ExtensionManager, ScalaConversions}
import org.nlogo.core.AgentKind
import scala.collection.mutable.ListBuffer
import scala.math.exp


class SimpleGeneticNeuralNet extends api.DefaultClassManager {

  class gNN(var in: Int = 0, var out: Int = 0, var layers: Int = 0) {

    var matrixLayers: Array[Array[Array[Double]]] = null
    var activationParameters: Array[Array[Array[Double]]] = null
    var convolutionMatrix: Array[Array[Double]] = null

    def initializeRandom(): Unit = {
      val r = scala.util.Random
      matrixLayers = Array.ofDim[Array[Array[Double]]](layers)
      for (x <- 0 until layers) {
        var mtrx = Array.ofDim[Double](in, in)

        for (i <- 0 until in) {
          for (j <- 0 until in) {
            mtrx(i)(j) = r.nextDouble()
          }
        }
        matrixLayers(x) = mtrx
      }
      activationParameters = Array.ofDim[Array[Array[Double]]](layers)
      for (x <- 0 until layers) {
        var mtrx = Array.ofDim[Double](in, 2)
        for (i <- 0 until out) {
          mtrx(i)(0) = r.nextDouble()
          val roll = r.nextInt(2)
          if (roll == 1) {
            mtrx(i)(1) = 1.0
          }
          else {
            mtrx(i)(1) = -1.0
          }
        }
        activationParameters(x) = mtrx
      }
      convolutionMatrix = Array.ofDim[Double](out, in)
      for (i <- 0 until out) {
        for (j <- 0 until in) {
          convolutionMatrix(i)(j) = r.nextDouble()
        }
      }
    }

    def swapRows(a: Int, b: Int, c: Int): Unit = {
      val temp = matrixLayers(c)(a)
      matrixLayers(c)(a) = matrixLayers(c)(b)
      matrixLayers(c)(b) = temp
    }

    def swapColumns(a: Int, b: Int, c: Int): Unit ={
      for (x <- 0 until in) {
        val temp = matrixLayers(c)(x)(a)
        matrixLayers(c)(x)(a) = matrixLayers(c)(x)(b)
        matrixLayers(c)(x)(b) = temp
      }

    }

    def mutateFromParent(other: gNN, rate: Double, range: Double): Unit = {
      in = other.in
      out = other.out
      layers = other.layers
      val r = scala.util.Random
      matrixLayers = Array.ofDim[Array[Array[Double]]](layers)
      for (x <- 0 until layers) {
        var mtrx = Array.ofDim[Double](in, in)
        for (i <- 0 until in) {
          for (j <- 0 until in) {
            val roll = r.nextDouble()
            if (roll < rate) {
              val roll2 = r.nextDouble()
              val roll3 = r.nextInt(2)
              if (roll3 == 1) {
                mtrx(i)(j) = other.matrixLayers(x)(i)(j) + roll2 * range
                if (mtrx(i)(j) < 0) {
                  mtrx(i)(j) = 0
                }
                if (mtrx(i)(j) > 1) {
                  mtrx(i)(j) = 1
                }
              }
              else {
                mtrx(i)(j) = other.matrixLayers(x)(i)(j) - roll2 * range
                if (mtrx(i)(j) < 0) {
                  mtrx(i)(j) = 0
                }
                if (mtrx(i)(j) > 1) {
                  mtrx(i)(j) = 1
                }
              }
            }
            else {
              mtrx(i)(j) = other.matrixLayers(x)(i)(j)
            }
          }
        }
        matrixLayers(x) = mtrx
      }
      activationParameters = Array.ofDim[Array[Array[Double]]](layers)
      for (x <- 0 until layers) {
        var mtrx = Array.ofDim[Double](in, 2)
        for (i <- 0 until out) {
          val roll = r.nextDouble()
          if (roll < rate) {
            val roll2 = r.nextDouble()
            val roll3 = r.nextInt(2)
            if (roll3 == 1) {
              mtrx(i)(0) = other.activationParameters(x)(i)(0) + roll2 * range
              if (mtrx(i)(0) < 0) {
                mtrx(i)(0) = 0
              }
              if (mtrx(i)(0) > 1) {
                mtrx(i)(0) = 1
              }
            }
            else {
              mtrx(i)(0) = other.activationParameters(x)(i)(0) - roll2 * range
              if (mtrx(i)(0) < 0) {
                mtrx(i)(0) = 0
              }
              if (mtrx(i)(0) > 1) {
                mtrx(i)(0) = 1
              }
            }
          }
          else {
            mtrx(i)(0) = other.activationParameters(x)(i)(0)
          }
          val roll4 = r.nextDouble()
          if (roll4 < rate) {
            mtrx(i)(1) = -1.0 * other.activationParameters(x)(i)(1)
          }
          else {
            mtrx(i)(1) = other.activationParameters(x)(i)(1)
          }
        }
        activationParameters(x) = mtrx
      }
      convolutionMatrix = Array.ofDim[Double](out, in)
      for (i <- 0 until out) {
        for (j <- 0 until in) {
          val roll = r.nextDouble()
          if (roll < rate) {
            val roll2 = r.nextDouble()
            val roll3 = r.nextInt(2)
            if (roll3 == 1) {
              convolutionMatrix(i)(j) = other.convolutionMatrix(i)(j) + roll2 * range
              if (convolutionMatrix(i)(j) < 0) {
                convolutionMatrix(i)(j) = 0
              }
              if (convolutionMatrix(i)(j) > 1) {
                convolutionMatrix(i)(j) = 1
              }
            }
            else {
              convolutionMatrix(i)(j) = other.convolutionMatrix(i)(j) - roll2 * range
              if (convolutionMatrix(i)(j) < 0) {
                convolutionMatrix(i)(j) = 0
              }
              if (convolutionMatrix(i)(j) > 1) {
                convolutionMatrix(i)(j) = 1
              }
            }
          }
          else {
            convolutionMatrix(i)(j) = other.convolutionMatrix(i)(j)
          }
        }
      }
      val roll5 = r.nextDouble()
      if (roll5 < (rate * 4)) {
        val a = r.nextInt(in)
        var b = r.nextInt(in)
        while (a == b) {b = r.nextInt(in)}
        val c = r.nextInt(layers)
        val roll6 = r.nextDouble()
        if (roll6 < 0.5) {
          swapColumns(a,b,c)

        } else {
          swapRows(a,b,c)
        }
      }
    }

    def choice(theVector: Array[Double]): Int = {
      var result = Array.ofDim[Double](in)
      var source = theVector.clone
      for (x <- 0 until layers) {
        for (i <- 0 until in) {
          var sum: Double = 0.0
          for (j <- 0 until in) {
            sum += source(j) * matrixLayers(x)(i)(j)
          }
          result(i) = sum
        }
        source = result.clone
        var max: Double = 0.0
        for (i <- 0 until in) {
          if (source(i) > max) {
            max = source(i)
          }
        }
        if (max > 0) {
          for (i <- 0 until in) {
            source(i) = source(i) / max
          }
        }

        for (i <- 0 until in) { // invert sigmoid
          if (activationParameters(x)(i)(1) == 1) {
            result(i) = (1 / (1.0 + exp(-50 * (source(i) - activationParameters(x)(i)(0)))))
          }
          else {
            result(i) = 1 - (1 / (1.0 + exp(-50 * (source(i) - activationParameters(x)(i)(0)))))
          }
        }
        source = result.clone
         }
        var finalResult = Array.ofDim[Double](out)
        for (i <- 0 until out) {
          var sum: Double = 0.0
          for (j <- 0 until in) {
            sum += source(j) * convolutionMatrix(i)(j)
          }
          finalResult(i) = sum
        }
        var total: Double = 0.0
        for (i <- 0 until out) {
          total += finalResult(i)
        }
        if (total > 0) {
          for (i <- 0 until out) {
            finalResult(i) = finalResult(i) / total
          }

          val r = new scala.util.Random(System.nanoTime())
          var roll = r.nextDouble()
          var count: Int = 0
          while (roll >= 0) {
            roll -= finalResult(count)
            count += 1
          }
          count
        }
        else {
          0
        }
        // return finalResult
      }
    }

  var turtlesToMatrices: mutable.Map[api.Turtle, gNN] = mutable.LinkedHashMap[api.Turtle, gNN]()
  var layers: Int = 3
  var in: Int = 6
  var out: Int = 3
  var mutationChance: Double = 0.005
  var mutationRange: Double = 0.10

  override def clearAll(): Unit = {
    super.clearAll()
    turtlesToMatrices = mutable.LinkedHashMap[api.Turtle, gNN]()
  }

  override def runOnce(em: ExtensionManager): Unit = {
    super.runOnce(em)
    clearAll()
  }

  def load(manager: api.PrimitiveManager) {
    manager.addPrimitive("random-brain", addMatrix)
    manager.addPrimitive("get-matrix", getMatrix)
    manager.addPrimitive("get-brain", getWholeNet)
    manager.addPrimitive("clear-brain", clearMatrix)
    manager.addPrimitive("brain-from-parent", matrixFromParent)
    manager.addPrimitive("make-choice", makeChoice)
    manager.addPrimitive("set-up", setUp)
    manager.addPrimitive("count-brains", countBrains)
    manager.addPrimitive("size-of-brains", sizeOfBrains)
    manager.addPrimitive("set-brain", setWholeNet)

  }

  object addMatrix extends api.Command {
    override def getSyntax: Syntax = Syntax.commandSyntax(right = List(), agentClassString = "-T--")

    override def perform(args: Array[Argument], context: Context): Unit = {
      var net = new gNN(in, out, layers)
      net.initializeRandom()
      context.getAgent match {
        case turtle: api.Turtle => turtlesToMatrices.update(turtle, net)
      }
    }
  }

  object getMatrix extends api.Reporter {
    override def getSyntax =
    Syntax.reporterSyntax(right = List(), ret = Syntax.ListType, agentClassString = "-T--")

    def report(args: Array[Argument], context: Context): AnyRef = {
      turtlesToMatrices(context.getAgent.asInstanceOf[api.Turtle]).matrixLayers.toLogoList
    }
  }

  object getWholeNet extends api.Reporter {
    override def getSyntax =
      Syntax.reporterSyntax(right = List(), ret = Syntax.ListType, agentClassString = "-T--")

    def report(args: Array[Argument], context: Context): AnyRef = {
      val theT = turtlesToMatrices(context.getAgent.asInstanceOf[api.Turtle])
      Array(Int.box(theT.layers), Int.box(theT.in), Int.box(theT.out), theT.matrixLayers, theT.activationParameters, theT.convolutionMatrix).toLogoList
    }
  }


  object setWholeNet extends api.Command {
    override def getSyntax =
      Syntax.commandSyntax(right = List(Syntax.ListType), agentClassString = "-T--")

    def perform(args: Array[api.Argument], context: api.Context): Unit = {
      var net = new gNN(in, out, layers)
      val list = args(0).getList
      val nLayers = list(0).asInstanceOf[Double].toInt
      val nIn = list(1).asInstanceOf[Double].toInt
      val nOut = list(2).asInstanceOf[Double].toInt
      val matrix = list(3).asInstanceOf[core.LogoList]
      val params = list(4).asInstanceOf[core.LogoList]
      val conv = list(5).asInstanceOf[core.LogoList]
      var matrixLayers = Array.ofDim[Array[Array[Double]]](nLayers)
      for (x <- 0 until nLayers) {
        val layer = matrix(x).asInstanceOf[core.LogoList]
        var mtrx = Array.ofDim[Double](nIn, nIn)
        for (i <- 0 until nIn) {
          val row = layer(i).asInstanceOf[core.LogoList]
          for (j <- 0 until nIn) {
            mtrx(i)(j) = row(j).asInstanceOf[Double]
          }
        }
        matrixLayers(x) = mtrx
      }
      var activationParams = Array.ofDim[Array[Array[Double]]](nLayers)
      for (x <- 0 until nLayers) {
        val layer = params(x).asInstanceOf[core.LogoList]
        var mtrx = Array.ofDim[Double](nIn, 2)
        for (i <- 0 until nIn) {
          val row = layer(i).asInstanceOf[core.LogoList]
          mtrx(i)(0) = row(0).asInstanceOf[Double]
          mtrx(i)(1) = row(1).asInstanceOf[Double]
        }
        activationParams(x) = mtrx
      }
      var conMatrix = Array.ofDim[Double](nOut, nIn)
      for (i <- 0 until nOut) {
        val row = conv(i).asInstanceOf[core.LogoList]
        for (j <- 0 until nIn){
          conMatrix(i)(j) = row(j).asInstanceOf[Double]
        }
      }

      net.matrixLayers = matrixLayers
      net.activationParameters = activationParams
      net.convolutionMatrix = conMatrix

      context.getAgent match {
        case turtle: api.Turtle => turtlesToMatrices.update(turtle, net)
      }

    }
  }


  object countBrains extends api.Reporter {
    override def getSyntax =
      Syntax.reporterSyntax(right = List(), ret = Syntax.NumberType, agentClassString = "O---")

    def report(args: Array[Argument], context: Context): AnyRef = {
      Double.box(turtlesToMatrices.size)
    }
  }



  object sizeOfBrains extends api.Reporter {
    override def getSyntax =
      Syntax.reporterSyntax(right = List(), ret = Syntax.NumberType, agentClassString = "O---")

    def report(args: Array[Argument], context: Context): AnyRef = {
      Double.box(12.0)
    }
  }


  object clearMatrix extends api.Command {
    override def getSyntax =
      Syntax.commandSyntax(right = List(), agentClassString = "-T--")

    def perform(args: Array[Argument], context: Context): Unit = {
      turtlesToMatrices.remove(context.getAgent.asInstanceOf[api.Turtle])
    }
  }

  object matrixFromParent extends api.Command {
    override def getSyntax =
      Syntax.commandSyntax(right = List(Syntax.TurtleType), agentClassString = "-T--")

    def perform(args: Array[Argument], context: Context): Unit = {
      var net = new gNN(3, 2, 2)
      var otherT: api.Turtle = null
      args(0).getAgent match {
        case turtle: api.Turtle => otherT = turtle
      }
      val parentNet = turtlesToMatrices.get(otherT)
      net.mutateFromParent(parentNet.get, mutationChance, mutationRange)
      context.getAgent match {
        case turtle: api.Turtle => turtlesToMatrices.update(turtle, net)
      }
    }
  }

  object makeChoice extends api.Reporter {
    override def getSyntax =
      Syntax.reporterSyntax(right = List(Syntax.ListType), ret = Syntax.NumberType, agentClassString = "-T--")

    def report(args: Array[api.Argument], context: api.Context): AnyRef = {
      val list = args(0).getList
      val ll = list.size
      var rl = Array.ofDim[Double](ll)
      for (i <- 0 until ll) {
        rl(i) = list(i).asInstanceOf[Double]
      }
      Double.box(turtlesToMatrices.get(context.getAgent.asInstanceOf[api.Turtle]).get.choice(rl))
    }
  }

  object setUp extends api.Command {
    override def getSyntax =
      Syntax.commandSyntax(right = List(Syntax.NumberType, Syntax.NumberType, Syntax.NumberType, Syntax.NumberType, Syntax.NumberType), agentClassString = "O---")

    def perform(args: Array[Argument], context: Context): Unit = {
      layers = args(0).getIntValue
      in = args(1).getIntValue
      out = args(2).getIntValue
      mutationChance = args(3).getDoubleValue
      mutationRange = args(4).getDoubleValue
    }
  }
}
