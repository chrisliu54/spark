/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.mllib.classification

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.classification.impl.GLMClassificationModel
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS.dot
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.{DataValidators, Loader, Saveable}
import org.apache.spark.rdd.RDD

class TransferLogisticRegressionModel (
                                     override val weights: Vector,
                                     override val intercept: Double,
                                     val numFeatures: Int,
                                     val numClasses: Int)
  extends TransferGeneralizedLinearModel(weights, intercept)
    with ClassificationModel with Serializable
    with Saveable with PMMLExportable {

  if (numClasses == 2) {
    require(weights.size == numFeatures,
      s"TransferLogisticRegressionModel with numClasses = 2 was given non-matching values:" +
        s" numFeatures = $numFeatures, but weights.size = ${weights.size}")
  } else {
    val weightsSizeWithoutIntercept = (numClasses - 1) * numFeatures
    val weightsSizeWithIntercept = (numClasses - 1) * (numFeatures + 1)
    require(weights.size == weightsSizeWithoutIntercept || weights.size == weightsSizeWithIntercept,
      s"TransferLogisticRegressionModel.load with numClasses = $numClasses and numFeatures = $numFeatures" +
        s" expected weights of length $weightsSizeWithoutIntercept (without intercept)" +
        s" or $weightsSizeWithIntercept (with intercept)," +
        s" but was given weights of length ${weights.size}")
  }

  private val dataWithBiasSize: Int = weights.size / (numClasses - 1)

  private val weightsArray: Array[Double] = weights match {
    case dv: DenseVector => dv.values
    case _ =>
      throw new IllegalArgumentException(
        s"weights only supports dense vector but got type ${weights.getClass}.")
  }

  /**
    * Constructs a [[TransferLogisticRegressionModel]] with weights and intercept for
    * binary classification.
    */
  def this(weights: Vector, intercept: Double) = this(weights, intercept, weights.size, 2)

  private var threshold: Option[Double] = Some(0.5)

  /**
    * Sets the threshold that separates positive predictions from negative predictions
    * in Binary Logistic Regression. An example with prediction score greater than or equal to
    * this threshold is identified as a positive, and negative otherwise. The default value is 0.5.
    * It is only used for binary classification.
    */
  def setThreshold(threshold: Double): this.type = {
    this.threshold = Some(threshold)
    this
  }

  /**
    * Returns the threshold (if any) used for converting raw prediction scores into 0/1 predictions.
    * It is only used for binary classification.
    */
  def getThreshold: Option[Double] = threshold

  /**
    * Clears the threshold so that `predict` will output raw prediction scores.
    * It is only used for binary classification.
    */
  def clearThreshold(): this.type = {
    threshold = None
    this
  }

  override protected def predictPoint(
                                       dataMatrix: Vector,
                                       weightMatrix: Vector,
                                       intercept: Double) = {
    require(dataMatrix.size == numFeatures)

    // If dataMatrix and weightMatrix have the same dimension, it's binary logistic regression.
    if (numClasses == 2) {
      val margin = dot(weightMatrix, dataMatrix) + intercept
      val score = 1.0 / (1.0 + math.exp(-margin))
      threshold match {
        case Some(t) => if (score > t) 1.0 else 0.0
        case None => score
      }
    } else {
      /**
        * Compute and find the one with maximum margins. If the maxMargin is negative, then the
        * prediction result will be the first class.
        *
        * PS, if you want to compute the probabilities for each outcome instead of the outcome
        * with maximum probability, remember to subtract the maxMargin from margins if maxMargin
        * is positive to prevent overflow.
        */
      var bestClass = 0
      var maxMargin = 0.0
      val withBias = dataMatrix.size + 1 == dataWithBiasSize
      (0 until numClasses - 1).foreach { i =>
        var margin = 0.0
        dataMatrix.foreachActive { (index, value) =>
          if (value != 0.0) margin += value * weightsArray((i * dataWithBiasSize) + index)
        }
        // Intercept is required to be added into margin.
        if (withBias) {
          margin += weightsArray((i * dataWithBiasSize) + dataMatrix.size)
        }
        if (margin > maxMargin) {
          maxMargin = margin
          bestClass = i + 1
        }
      }
      bestClass.toDouble
    }
  }

  override def save(sc: SparkContext, path: String): Unit = {
    GLMClassificationModel.SaveLoadV1_0.save(sc, path, this.getClass.getName,
      numFeatures, numClasses, weights, intercept, threshold)
  }

  override protected def formatVersion: String = "1.0"

  override def toString: String = {
    s"${this.getClass.getName}: intercept = ${intercept}, numFeatures = ${weights.size}," +
      s"numClasses = ${numClasses}, threshold = ${threshold.getOrElse("None")}"
  }

}

object TransferLogisticRegressionModel extends Loader[TransferLogisticRegressionModel] {

  override def load(sc: SparkContext, path: String): TransferLogisticRegressionModel = {
    val (loadedClassName, version, metadata) = Loader.loadMetadata(sc, path)
    // Hard-code class name string in case it changes in the future
    val classNameV1_0 = "org.apache.spark.mllib.classification.TransferLogisticRegressionModel"
    (loadedClassName, version) match {
      case (className, "1.0") if className == classNameV1_0 =>
        val (numFeatures, numClasses) = ClassificationModel.getNumFeaturesClasses(metadata)
        val data = GLMClassificationModel.SaveLoadV1_0.loadData(sc, path, classNameV1_0)
        // numFeatures, numClasses, weights are checked in model initialization
        val model =
          new TransferLogisticRegressionModel(data.weights, data.intercept, numFeatures, numClasses)
        data.threshold match {
          case Some(t) => model.setThreshold(t)
          case None => model.clearThreshold()
        }
        model
      case _ => throw new Exception(
        s"TransferLogisticRegressionModel.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $version).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)")
    }
  }
}

class TransferLogisticRegressionWithSGD private[mllib] (
                                                      private var stepSize: Double,
                                                      private var numIterations: Int,
                                                      private var regParam: Double,
                                                      private var miniBatchFraction: Double)
  extends TransferGeneralizedLinearAlgorithm[TransferLogisticRegressionModel] with Serializable {

  private val gradient = new LogisticGradient()
  private val updater = new SquaredL2Updater()
  @Since("0.8.0")
  override val optimizer = new TransferGradientDescent(gradient, updater)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setRegParam(regParam)
    .setMiniBatchFraction(miniBatchFraction)
  override protected val validators = List(DataValidators.binaryLabelValidator)

  /**
    * Construct a TransferLogisticRegression object with default parameters: {stepSize: 1.0,
    * numIterations: 100, regParm: 0.01, miniBatchFraction: 1.0}.
    */
  def this() = this(1.0, 100, 0.01, 1.0)

  override protected[mllib] def createModel(weights: Vector, intercept: Double) = {
    new TransferLogisticRegressionModel(weights, intercept)
  }
}

/**
  * Top-level methods for calling Transfer Logistic Regression using Stochastic Gradient Descent.
  *
  * @note Labels used in Logistic Regression should be {0, 1}
  */
object TransferLogisticRegressionWithSGD {
  // NOTE(shivaram): We use multiple train methods instead of default arguments to support
  // Java programs.

  /**
    * Train a transfer logistic regression model given an RDD of (label, features) pairs and
    * another RDD of Vector. We run a fixed number of iterations of gradient descent using
    * the specified step size. Each iteration uses `miniBatchFraction` fraction of the data
    * to calculate the gradient. The weights used in gradient descent are initialized using
    * the initial weights provided.
    *
    * @param labeledInput RDD of (label, array of features) pairs.
    * @param unlabeledInput RDD of features Vector.
    * @param numIterations Number of iterations of gradient descent to run.
    * @param stepSize Step size to be used for each iteration of gradient descent.
    * @param miniBatchFraction Fraction of data to be used per iteration.
    * @param initialWeights Initial set of weights to be used. Array should be equal in size to
    *        the number of features in the data.
    *
    * @note Labels used in Logistic Regression should be {0, 1}
    */
  def train(
             labeledInput: RDD[LabeledPoint],
             unlabeledInput: RDD[Vector],
             numIterations: Int,
             stepSize: Double,
             miniBatchFraction: Double,
             initialWeights: Vector): TransferLogisticRegressionModel = {
    new TransferLogisticRegressionWithSGD(stepSize, numIterations, 0.0, miniBatchFraction)
      .run(labeledInput, unlabeledInput, initialWeights)
  }

  /**
    * Train a transfer logistic regression model given an RDD of (label, features) pairs
    * and another RDD of Vector. We run a fixed number of iterations of gradient descent
    * using the specified step size. Each iteration uses `miniBatchFraction` fraction
    * of the data to calculate the gradient.
    *
    * @param labeledInput RDD of (label, array of features) pairs.
    * @param unlabeledInput RDD of features Vector.
    * @param numIterations Number of iterations of gradient descent to run.
    * @param stepSize Step size to be used for each iteration of gradient descent.
    * @param miniBatchFraction Fraction of data to be used per iteration.
    *
    * @note Labels used in Logistic Regression should be {0, 1}
    */
  def train(
             labeledInput: RDD[LabeledPoint],
             unlabeledInput: RDD[Vector],
             numIterations: Int,
             stepSize: Double,
             miniBatchFraction: Double): TransferLogisticRegressionModel = {
    new TransferLogisticRegressionWithSGD(stepSize, numIterations, 0.0, miniBatchFraction)
      .run(labeledInput, unlabeledInput)
  }

  /**
    * Train a transfer logistic regression model given an RDD of (label, features) pairs
    * and another RDD of Vector. We run a fixed number of iterations of gradient descent
    * using the specified step size. We use the entire data set to update the gradient
    * in each iteration.
    *
    * @param labeledInput RDD of (label, array of features) pairs.
    * @param unlabeledInput RDD of features Vector.
    * @param stepSize Step size to be used for each iteration of Gradient Descent.
    * @param numIterations Number of iterations of gradient descent to run.
    * @return a LogisticRegressionModel which has the weights and offset from training.
    *
    * @note Labels used in Logistic Regression should be {0, 1}
    */
  def train(
             labeledInput: RDD[LabeledPoint],
             unlabeledInput: RDD[Vector],
             numIterations: Int,
             stepSize: Double): TransferLogisticRegressionModel = {
    train(labeledInput, unlabeledInput, numIterations, stepSize, 1.0)
  }

  /**
    * Train a transfer logistic regression model given an RDD of (label, features) pairs
    * and another RDD of Vector. We run a fixed number of iterations of gradient descent
    * using a step size of 1.0. We use the entire data set to update the gradient
    * in each iteration.
    *
    * @param labeledInput RDD of (label, array of features) pairs.
    * @param unlabeledInput RDD of features Vector.
    * @param numIterations Number of iterations of gradient descent to run.
    * @return a LogisticRegressionModel which has the weights and offset from training.
    *
    * @note Labels used in Logistic Regression should be {0, 1}
    */
  def train(
             labeledInput: RDD[LabeledPoint],
             unlabeledInput: RDD[Vector],
             numIterations: Int): TransferLogisticRegressionModel = {
    train(labeledInput, unlabeledInput, numIterations, 1.0, 1.0)
  }
}