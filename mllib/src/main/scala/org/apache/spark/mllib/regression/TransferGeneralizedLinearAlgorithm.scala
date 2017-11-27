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

package org.apache.spark.mllib.regression

import org.apache.spark.SparkException
import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
  * :: DeveloperApi ::
  * TransferGeneralizedLinearModel (TGLM) represents a model trained using
  * TransferGeneralizedLinearAlgorithm.
  * GLMs consist of a weight vector and an intercept.
  *
  * @param weights Weights computed for every feature.
  * @param intercept Intercept computed for this model.
  *
  */
@DeveloperApi
abstract class TransferGeneralizedLinearModel
(val weights: Vector, val intercept: Double)
  extends Serializable {

  /**
    * Predict the result given a data point and the weights learned.
    *
    * @param dataMatrix Row vector containing the features for this data point
    * @param weightMatrix Column vector containing the weights of the model
    * @param intercept Intercept of the model.
    */
  protected def predictPoint(dataMatrix: Vector, weightMatrix: Vector, intercept: Double): Double

  /**
    * Predict values for the given data set using the model trained.
    *
    * @param testData RDD representing data points to be predicted
    * @return RDD[Double] where each entry contains the corresponding prediction
    *
    */
  def predict(testData: RDD[Vector]): RDD[Double] = {
    // A small optimization to avoid serializing the entire model. Only the weightsMatrix
    // and intercept is needed.
    val localWeights = weights
    val bcWeights = testData.context.broadcast(localWeights)
    val localIntercept = intercept
    testData.mapPartitions { iter =>
      val w = bcWeights.value
      iter.map(v => predictPoint(v, w, localIntercept))
    }
  }

  /**
    * Predict values for a single data point using the model trained.
    *
    * @param testData array representing a single data point
    * @return Double prediction from the trained model
    *
    */
  def predict(testData: Vector): Double = {
    predictPoint(testData, weights, intercept)
  }

  /**
    * Print a summary of the model.
    */
  override def toString: String = {
    s"${this.getClass.getName}: intercept = ${intercept}, numFeatures = ${weights.size}"
  }
}

/**
  * :: DeveloperApi ::
  * TransferGeneralizedLinearAlgorithm implements methods to train
  * a Transfer Generalized Linear Model (TGLM).
  * This class should be extended with an Optimizer to create a new TGLM.
  *
  */
@DeveloperApi
abstract class TransferGeneralizedLinearAlgorithm[M <: TransferGeneralizedLinearModel]
  extends Logging with Serializable {

  protected val validators: Seq[RDD[LabeledPoint] => Boolean] = List()

  /**
    * The optimizer to solve transfer learning problem.
    */
  def optimizer: TransferOptimizer

  /** Whether to add intercept (default: false). */
  protected var addIntercept: Boolean = false

  protected var validateData: Boolean = true

  /**
    * In `GeneralizedLinearModel`, only single linear predictor is allowed for both weights
    * and intercept. However, for multinomial logistic regression, with K possible outcomes,
    * we are training K-1 independent binary logistic regression models which requires K-1 sets
    * of linear predictor.
    *
    * As a result, the workaround here is if more than two sets of linear predictors are needed,
    * we construct bigger `weights` vector which can hold both weights and intercepts.
    * If the intercepts are added, the dimension of `weights` will be
    * (numOfLinearPredictor) * (numFeatures + 1) . If the intercepts are not added,
    * the dimension of `weights` will be (numOfLinearPredictor) * numFeatures.
    *
    * Thus, the intercepts will be encapsulated into weights, and we leave the value of intercept
    * in GeneralizedLinearModel as zero.
    */
  protected var numOfLinearPredictor: Int = 1

  /**
    * Whether to perform feature scaling before model training to reduce the condition numbers
    * which can significantly help the optimizer converging faster. The scaling correction will be
    * translated back to resulting model weights, so it's transparent to users.
    * Note: This technique is used in both libsvm and glmnet packages. Default false.
    */
  private[mllib] var useFeatureScaling = false

  /**
    * The dimension of training features.
    *
    */
  def getNumFeatures: Int = this.numFeatures

  /**
    * The dimension of training features.
    */
  protected var numFeatures: Int = -1

  /**
    * Set if the algorithm should use feature scaling to improve the convergence during optimization.
    */
  private[mllib] def setFeatureScaling(useFeatureScaling: Boolean): this.type = {
    this.useFeatureScaling = useFeatureScaling
    this
  }

  /**
    * Create a model given the weights and intercept
    */
  protected def createModel(weights: Vector, intercept: Double): M

  /**
    * Get if the algorithm uses addIntercept
    *
    */
  @Since("1.4.0")
  def isAddIntercept: Boolean = this.addIntercept

  /**
    * Set if the algorithm should add an intercept. Default false.
    * We set the default to false because adding the intercept will cause memory allocation.
    *
    */
  @Since("0.8.0")
  def setIntercept(addIntercept: Boolean): this.type = {
    this.addIntercept = addIntercept
    this
  }

  /**
    * Set if the algorithm should validate data before training. Default true.
    *
    */
  @Since("0.8.0")
  def setValidateData(validateData: Boolean): this.type = {
    this.validateData = validateData
    this
  }

  /**
    * Generate the initial weights when the user does not supply them
    */
  protected def generateInitialWeights(input: RDD[LabeledPoint]): Vector = {
    if (numFeatures < 0) {
      numFeatures = input.map(_.features.size).first()
    }

    /**
      * When `numOfLinearPredictor > 1`, the intercepts are encapsulated into weights,
      * so the `weights` will include the intercepts. When `numOfLinearPredictor == 1`,
      * the intercept will be stored as separated value in `GeneralizedLinearModel`.
      * This will result in different behaviors since when `numOfLinearPredictor == 1`,
      * users have no way to set the initial intercept, while in the other case, users
      * can set the intercepts as part of weights.
      *
      * TODO: See if we can deprecate `intercept` in `GeneralizedLinearModel`, and always
      * have the intercept as part of weights to have consistent design.
      */
    if (numOfLinearPredictor == 1) {
      Vectors.zeros(numFeatures)
    } else if (addIntercept) {
      Vectors.zeros((numFeatures + 1) * numOfLinearPredictor)
    } else {
      Vectors.zeros(numFeatures * numOfLinearPredictor)
    }
  }

  /**
    * This is for TGLM.
    * Run the algorithm with the configured parameters on an RDD
    * of LabeledPoint entries and another RDD of Vector entries
    * from the initial weights provided.
    * @param labeledInput
    * @param unlabeledInput
    * @return
    */
  def run(labeledInput: RDD[LabeledPoint], unlabeledInput: RDD[Vector]): M = {
    run(labeledInput, unlabeledInput, generateInitialWeights(labeledInput))
  }

  def run(labeledInput: RDD[LabeledPoint], unlabeledInput: RDD[Vector],
          initialWeights: Vector): M = {
    if (numFeatures < 0) {
      numFeatures = labeledInput.map(_.features.size).first()
    }

    if (labeledInput.getStorageLevel == StorageLevel.NONE
      || unlabeledInput.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Check the input_labeled data properties before running the optimizer
    if (validateData && !validators.forall(func => func(labeledInput))) {
      throw new SparkException("Labeled input validation failed.")
    }

    // create `scaler` using both the labeled data and the unlabeled
    val scaler = if (useFeatureScaling) {
      val trainingData = labeledInput.map(_.features).union(unlabeledInput)
      new StandardScaler(withStd = true, withMean = false).fit(trainingData)
    } else {
      null
    }

    // Prepend an extra variable consisting of all 1.0's for the intercept.
    // TODO: Apply feature scaling to the weight vector instead of input data.
    val labeledData =
    if (addIntercept) {
      if (useFeatureScaling) {
        labeledInput.map(lp => (lp.label, appendBias(scaler.transform(lp.features)))).cache()
      } else {
        labeledInput.map(lp => (lp.label, appendBias(lp.features))).cache()
      }
    } else {
      if (useFeatureScaling) {
        labeledInput.map(lp => (lp.label, scaler.transform(lp.features))).cache()
      } else {
        labeledInput.map(lp => (lp.label, lp.features))
      }
    }

    val unlabeledData =
      if (addIntercept) {
        if (useFeatureScaling) {
          unlabeledInput.map(v => appendBias(scaler.transform(v))).cache()
        } else {
          unlabeledInput.map(v => appendBias(v)).cache()
        }
      } else {
        if (useFeatureScaling) {
          unlabeledInput.map(v => scaler.transform(v)).cache()
        } else {
          unlabeledInput.cache()
        }
      }


    val initialWeightsWithIntercept = if (addIntercept && numOfLinearPredictor == 1) {
      appendBias(initialWeights)
    } else {
      /** If `numOfLinearPredictor > 1`, initialWeights already contains intercepts. */
      initialWeights
    }

    val weightsWithIntercept = optimizer.optimize(labeledData,
      unlabeledData, initialWeightsWithIntercept)

    val intercept = if (addIntercept && numOfLinearPredictor == 1) {
      weightsWithIntercept(weightsWithIntercept.size - 1)
    } else {
      0.0
    }

    var weights = if (addIntercept && numOfLinearPredictor == 1) {
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1))
    } else {
      weightsWithIntercept
    }

    // Transform the learned weights using the previous `scaler`
    if (useFeatureScaling) {
      if (numOfLinearPredictor == 1) {
        weights = scaler.transform(weights)
      } else {
        var i = 0
        val n = weights.size / numOfLinearPredictor
        val weightsArray = weights.toArray
        while (i < numOfLinearPredictor) {
          val start = i * n
          val end = (i + 1) * n - { if (addIntercept) 1 else 0 }

          val partialWeightsArray = scaler.transform(
            Vectors.dense(weightsArray.slice(start, end))).toArray

          System.arraycopy(partialWeightsArray, 0, weightsArray, start, partialWeightsArray.length)
          i += 1
        }
        weights = Vectors.dense(weightsArray)
      }
    }

    // Warn at the end of the run as well, for increased visibility.
    if (labeledInput.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Unpersist cached data
    if (labeledData.getStorageLevel != StorageLevel.NONE) {
      labeledData.unpersist(false)
    }
    if (unlabeledData.getStorageLevel != StorageLevel.NONE) {
      unlabeledData.unpersist(false)
    }

    createModel(weights, intercept)
  }
}
