// Databricks notebook source
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

// data from http://www.cs.waikato.ac.nz/ml/weka/datasets.html (datasets-numeric/zoo)
// Load  data to DataFrame.
val data = spark.read.format("libsvm").load("/FileStore/tables/pu9qim6x1498342511430/YearPredictionMSD.libsvm")

// identify categorical features, features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Split data into 70% training and 30% test
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// build classifier
val rf = new RandomForestRegressor()
  .setLabelCol("label")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(10)
  .setMaxDepth(5)
  .setImpurity("variance")

// build pipeline
val pipeline = new Pipeline()
  .setStages(Array(featureIndexer, rf))

// Training
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display
predictions.select("prediction", "label", "features").show(20)

// Select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
println("Learned regression forest model:\n" + rfModel.toDebugString)
